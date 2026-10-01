// src/core/verify.cpp - see include/strata/core/verify.hpp.
#include "strata/core/verify.hpp"
#if defined(_WIN32)
#include <intrin.h>
#endif

#include "strata/core/native_head.hpp"
#include "strata/core/placement.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/cvec.hpp"
#include "strata/kernels/gr.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/ple.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/rope.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/core/progress.hpp"
#include "strata/kernels/shared_expert.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <immintrin.h>

namespace strata::core {
namespace {

constexpr float EPS = 1e-6f;
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }
const bool g_dbg = std::getenv("STRATA_VERIFY_DEBUG") != nullptr;
#define VDBG(...) do { if (g_dbg) { std::fprintf(stderr, "verify dbg: " __VA_ARGS__); std::fflush(stderr); } } while (0)

struct Bump {
    uint8_t* base = nullptr;
    uint64_t used = 0;
    template <typename T> T* take(uint64_t n) {
        T* p = base ? (T*) (base + used) : nullptr;
        used += (n * sizeof(T) + 255) & ~255ull;
        return p;
    }
};

// Portable: every GPU of the pipeline reads and writes the same staging.
bool mapped(size_t bytes, void** h, void** d) {
    if (cudaHostAlloc(h, bytes, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) return false;
    std::memset(*h, 0, bytes);
    return cudaHostGetDevicePointer(d, *h, 0) == cudaSuccess;
}

strata::kernels::QsaShapes shapes_of(const ModelGeometry& g) {
    strata::kernels::QsaShapes s = strata::kernels::qsa_real_shapes();
    s.n_head = g.n_head;
    s.n_head_kv = g.n_head_kv;
    s.head_dim = g.head_dim;
    s.idx_n_head = g.idx_q_heads;
    s.idx_dim = g.idx_key_dim;
    return s;
}

const WeightRef* need(const LayerView& v, const char* suffix, std::string& err) {
    const WeightRef* r = v.get(suffix);
    if (r == nullptr && err.empty()) err = v.name(suffix) + " is missing";
    return r;
}

bool native_of(const WeightRef* w, const std::string& name, std::string& err) {
    if (w == nullptr) return false;
    if (w->native_data == nullptr) {
        err = "verify: " + name + " is not served natively (run with --native)";
        return false;
    }
    return true;
}

}  // namespace

namespace {
std::atomic<const Verifier*> g_diag_verifier{nullptr};
void diag_active_verifier(std::FILE* f) {
    if (const Verifier* v = g_diag_verifier.load()) v->diag(f);
}
}  // namespace

void Verifier::diag(std::FILE* f) const {
    auto rd = [](const uint32_t* p) { return p ? *(const volatile uint32_t*) p : 0u; };
    std::fprintf(f, "  verify window: %d tokens at position %lld, host at layer step %u; the GPU rang %u; flags: "
                    "served %u, plan (A) %u, copies (B) %u", last_t_, (long long) last_pos0_, cur_layer_ + 1,
                 rd(h_seq_), rd(h_flag_), rd(h_flagA_), rd(h_flagB_));
    if (ep_) std::fprintf(f, ", first GPU's plan %u, its rows %u", rd(h_epA_), rd(h_epB_));
    for (size_t s = 0; s < h_hand_flag_.size(); ++s)
        std::fprintf(f, ", stage %zu handed on %u", s, rd(h_hand_flag_[s]));
    std::fprintf(f, "\n");
}

Verifier::~Verifier() {
    const Verifier* self = this;
    g_diag_verifier.compare_exchange_strong(self, nullptr);
    for (Part& P : stages_) {
        DeviceGuard dg(P.device);
        if (P.cs) cudaStreamSynchronize(P.cs);
        for (auto& e : P.exec)
            if (e) cudaGraphExecDestroy(e);
        if (P.commit_exec) cudaGraphExecDestroy(P.commit_exec);
        if (P.cs) cudaStreamDestroy(P.cs);
        if (P.copy) { cudaStreamSynchronize(P.copy); cudaStreamDestroy(P.copy); }
        if (P.side) cudaStreamDestroy(P.side);
        if (P.fork) cudaEventDestroy(P.fork);
        if (P.join) cudaEventDestroy(P.join);
        if (P.arena) cudaFree(P.arena);
    }
    void* hosts[] = {h_tok_, h_step_, h_pos_, h_pos_kv_, h_pos_iq_, h_commit_, h_ple_, h_out_, h_x_, h_ids_, h_w_, h_seq_, h_flag_, h_ymiss_,
                     h_flagA_, h_plan_, h_flagB_, h_eplan_, h_epA_, h_epB_};
    for (void* h : hosts)
        if (h) cudaFreeHost(h);
    for (void* h : h_hand_) cudaFreeHost(h);
    for (void* h : h_hand_flag_) cudaFreeHost(h);
}

bool Verifier::init(const WeightTable& wt, const ModelGeometry& g, SessionState& ss, const VerifyHits& hits,
                    const NativeHead* head, int max_t, std::string& err) {
    g_diag_verifier.store(this);
    diag_verify_fn().store(&diag_active_verifier);
    wt_ = &wt;
    g_ = &g;
    ss_ = &ss;
    hits_ = hits;
    head_ = head;
    max_t_ = max_t;
    sampling_.greedy = true;      // a fresh verifier samples greedily until set_sampling says otherwise
    sampling_.temperature = 0.0f;
    if (max_t < 2 || max_t > strata::kernels::kVerifyMaxT || max_t > strata::kernels::cpu::MAXT) {
        err = "verify: the window must hold 2.." + std::to_string(strata::kernels::kVerifyMaxT) + " tokens";
        return false;
    }
    if (hits.d_res == nullptr || hits.cache_base == nullptr || hits.blob <= 0) {
        err = "verify: needs the profile-filled VRAM expert tier (--expert-profile and --expert-cache); with "
              "--expert-cache auto, no VRAM was left for it: lower --max-context or use --kv k8v4";
        return false;
    }
    std::string why;
    if (!layer_verify_compatible(why)) {
        err = "verify: " + why + " (the verify window reproduces the default native decode path)";
        return false;
    }
    if (!strata::kernels::fused_gr_supported(g.n_embd, g.hc, g.hc_lr) || ss.k != 10 || g.ssm_state_size != 128 ||
        g.ssm_d_conv != 4) {
        err = "verify: geometry differs from the artifact's";
        return false;
    }
    const WeightRef* wo = wt.find("output.weight");
    if (wo == nullptr) { err = "verify: output.weight is missing"; return false; }
    n_vocab_ = wo->ne1;

    const strata::kernels::QsaShapes s = shapes_of(g);
    cap_ = strata::kernels::qsa_selection_width(strata::kernels::kTopkMaxCells, s);
    max_blocks_ = ss.qsa_states[0].max_cells / s.idx_block + 2;
    attn_scratch_floats_ = (int64_t) strata::kernels::qsa_decode_attn_scratch_floats(cap_, s);

    const uint64_t T = (uint64_t) max_t, N = (uint64_t) g.n_embd, HC = (uint64_t) g.hc, K = (uint64_t) ss.k;
    const uint64_t C = (uint64_t) g.ssm_conv_channels, ZV = (uint64_t) g.ssm_value_dim, HV = (uint64_t) g.ssm_v_heads;
    const uint64_t NH = (uint64_t) g.n_head, HD = (uint64_t) g.head_dim, NKV = (uint64_t) g.n_head_kv;
    const uint64_t IQ = (uint64_t) g.idx_q_heads, ID = (uint64_t) g.idx_key_dim;
    const uint64_t HS = (uint64_t) strata::kernels::NG_HIST * strata::kernels::NG_HC_DIM;
    const uint64_t TS = (uint64_t) (s.idx_block - 1) * ID;
    const int max_in = (int) std::max<uint64_t>(std::max<uint64_t>(N, ZV), NH * HD);

    // ---- mapped staging
    bool ok = mapped(T * 4, (void**) &h_tok_, (void**) &m_tok_) &&
              mapped(T * strata::kernels::kStepCount * 4, (void**) &h_step_, (void**) &m_step_) &&
              mapped(T * NH * 4, (void**) &h_pos_, (void**) &m_pos_) &&
              mapped(T * NKV * 4 + 16, (void**) &h_pos_kv_, (void**) &m_pos_kv_) &&
              mapped(T * IQ * 4 + 16, (void**) &h_pos_iq_, (void**) &m_pos_iq_) &&
              mapped((2 + T) * 4 + 16, (void**) &h_commit_, (void**) &m_commit_) &&
              mapped(T * N * 4, (void**) &h_ple_, (void**) &m_ple_) &&
              mapped(T * 4 + 16, (void**) &h_out_, (void**) &m_out_) &&
              mapped(T * N * 4, (void**) &h_x_, (void**) &m_x_) &&
              mapped(T * K * 4, (void**) &h_ids_, (void**) &m_ids_) &&
              mapped(T * K * 4, (void**) &h_w_, (void**) &m_w_) &&
              mapped(64, (void**) &h_seq_, (void**) &m_seq_) &&
              mapped(64, (void**) &h_flag_, (void**) &m_flag_) &&
              mapped(64, (void**) &h_flagA_, (void**) &m_flagA_) &&
              mapped(64, (void**) &h_flagB_, (void**) &m_flagB_) &&
              mapped(T * K * N * 4, (void**) &h_ymiss_, (void**) &m_ymiss_);
    if (!ok) { err = "verify: mapped staging allocation failed"; return false; }
    // the GPU plan: counts(4) | start(cap+1) | dst(cap) | tok(cap) | pad | ptr(cap u64) | ptr2(cap u64) | start2(cap+1)
    {
        const int64_t cap = (int64_t) (T * K);
        const int64_t i32 = 4 + (cap + 1) + cap + cap;
        const int64_t ptr_off = (i32 + 1) & ~1ll;
        plan_i32_ = ptr_off + 4 * cap + (cap + 1) + 1;
        if (!mapped((size_t) plan_i32_ * 4 * 2 + 64, (void**) &h_plan_, (void**) &m_plan_) ||
            !mapped((size_t) plan_i32_ * 4 * 2 + 64, (void**) &h_eplan_, (void**) &m_eplan_) ||
            !mapped(64, (void**) &h_epA_, (void**) &m_epA_) || !mapped(64, (void**) &h_epB_, (void**) &m_epB_)) {
            err = "verify: mapped plan allocation failed";
            return false;
        }
        sink_.counts = h_plan_;
        sink_.start = h_plan_ + 4;
        sink_.dst = sink_.start + cap + 1;
        sink_.tok = sink_.dst + cap;
        sink_.ptr = (unsigned long long*) (h_plan_ + ptr_off);
        sink_.ptr2 = sink_.ptr + cap;
        sink_.start2 = h_plan_ + ptr_off + 4 * cap;
        sink_.cap = cap;
        sink_.publish = &Verifier::publish_plan;
        sink_.fetch = &Verifier::fetch_dma;
        sink_.ctx = this;
    }

    // ---- the pipeline's stages (placement.hpp), each with the residual it hands on
    const Placement& pl = placement();
    stages_.assign((size_t) pl.stages(), Part{});
    {
        int64_t gi = 0, qi = 0;
        for (int st = 0; st < pl.stages(); ++st) {
            Part& P = stages_[(size_t) st];
            P.stage = st;
            P.device = pl.device(st);
            P.l0 = pl.first(st);
            P.l1 = pl.end(st);
            P.gdn0 = gi;
            P.qsa0 = qi;
            for (int64_t l = P.l0; l < P.l1; ++l) {
                if (is_qsa_layer(g, l)) ++qi;
                else ++gi;
            }
            P.n_gdn = gi - P.gdn0;
            P.n_qsa = qi - P.qsa0;
        }
    }
    h_hand_.assign(stages_.size() - 1, nullptr);
    m_hand_.assign(stages_.size() - 1, nullptr);
    h_hand_flag_.assign(stages_.size() - 1, nullptr);
    m_hand_flag_.assign(stages_.size() - 1, nullptr);
    for (size_t s = 0; s + 1 < stages_.size(); ++s)
        if (!mapped(T * HC * N * 4, (void**) &h_hand_[s], (void**) &m_hand_[s]) ||
            !mapped(64, (void**) &h_hand_flag_[s], (void**) &m_hand_flag_[s])) {
            err = "verify: mapped hand-off allocation failed";
            return false;
        }

    // ---- each stage's device arena: the same sequence counted, then carved
    auto carve = [&](Part& P, Bump& b) {
        const uint64_t pG = (uint64_t) P.n_gdn, pQ = (uint64_t) P.n_qsa;
        const bool last = P.stage == (int) stages_.size() - 1, ple = P.l0 <= 1 && 1 < P.l1;
        P.tok_ = b.take<int32_t>(T); P.step_ = b.take<int32_t>(T * strata::kernels::kStepCount);
        P.pos_ = b.take<int32_t>(T * NH); P.commit_ = b.take<int32_t>(2 + T);
        P.pos_kv_ = b.take<int32_t>(T * NKV); P.pos_iq_ = b.take<int32_t>(T * IQ);
        P.ple_ = b.take<float>(T * N); P.emb_ = b.take<float>(T * N); P.R_ = b.take<float>(T * HC * N);
        P.mixed_ = b.take<float>(T * N); P.bo_ = b.take<float>(T * N);
        P.inj_ = b.take<float>(T * HC); P.inj2_ = b.take<float>(T * HC);
        P.lo_ = b.take<float>(T * (uint64_t) g.hc_lr); P.rs_ = b.take<float>(T * HC); P.xn_ = b.take<float>(strata::kernels::fused_gr_multi_scratch_floats((int) T));
        P.xq_ = b.take<uint8_t>(strata::kernels::native_q8_1_bytes(max_in, (int) T));
        P.qkv_L_ = b.take<float>(pG * T * C); P.h_L_ = b.take<float>(pG * T * C);
        P.gate_L_ = b.take<float>(pG * T * HV); P.beta_L_ = b.take<float>(pG * T * HV);
        P.z_ = b.take<float>(T * ZV); P.y_ = b.take<float>(T * ZV); P.y_dummy_ = b.take<float>(T * ZV);
        P.qfull_ = b.take<float>(T * NH * 2 * HD); P.qcur_ = b.take<float>(T * NH * HD);
        P.kcur_ = b.take<float>(T * NKV * HD); P.vcur_ = b.take<float>(T * NKV * HD);
        P.idx_raw_L_ = b.take<float>(pQ * T * ID); P.qidx_ = b.take<float>(T * IQ * ID);
        P.scores_ = b.take<float>(T * (uint64_t) max_blocks_); P.sel_ = b.take<int32_t>(T * (uint64_t) cap_);
        P.attn_ = b.take<float>(T * NH * HD); P.attn32_ = b.take<float>(T * NH * HD);
        P.attn_scratch_ = b.take<float>(T * (uint64_t) attn_scratch_floats_);
        P.tail_snap_ = b.take<float>(pQ * TS);
        P.logits_ = b.take<float>(T * (uint64_t) g.n_expert); P.w_ = b.take<float>(T * K); P.ids_ = b.take<int32_t>(T * K);
        P.shared_ = b.take<float>(T * N); P.parts_ = b.take<float>(T * K * N); P.hit_out_ = b.take<float>(T * K * N);
        P.hit_slot_ = b.take<int32_t>(T * K); P.hit_dst_ = b.take<int32_t>(T * K); P.hit_count_ = b.take<int32_t>(4);
        P.plan_ = b.take<int32_t>(2 * ((uint64_t) plan_i32_ + 16));
        P.staging_ = b.take<uint8_t>((uint64_t) kStagingBlobs * strata::kernels::cpu::expert_layout().max_blob);
        P.hit_xq_ = b.take<uint8_t>(T * (N / 32) * 34); P.hit_xs_ = b.take<float>(T * (N / 32));
        P.nat_xq_ = b.take<uint8_t>(T * (N / 32) * 36);
        P.hit_scratch_ = b.take<uint8_t>(std::max<uint64_t>(
            strata::kernels::moe_hit_grouped_scratch_bytes((int64_t) (T * K), g.n_embd, g.n_ff),
            strata::kernels::native_expert_scratch_bytes((int64_t) (T * K), g.n_ff)));
        P.head_mixed_ = b.take<float>(T * N); P.head_inj_ = b.take<float>(HC);
        P.sh_bf16_ = b.take<uint16_t>(T * N); P.sh_gate_ = b.take<float>(T * (uint64_t) g.n_ff);
        P.sh_up_ = b.take<float>(T * (uint64_t) g.n_ff); P.sh_g_ = b.take<float>(T + 4);
        P.head_logits_ = b.take<float>(last ? T * (uint64_t) n_vocab_ : 0);
        P.hist_snap_ = b.take<float>(ple ? T * HS : 0);
        P.ple_key_ = b.take<float>(ple ? T * (uint64_t) strata::kernels::NG_HC_DIM : 0);
        P.ple_val_ = b.take<float>(ple ? T * N : 0);
    };
    uint64_t total = 0;
    for (Part& P : stages_) {
        DeviceGuard dg(P.device);
        Bump count;
        carve(P, count);
        if (cudaMalloc(&P.arena, count.used) != cudaSuccess) {
            err = "verify: the device arena (" + std::to_string(count.used >> 20) + " MiB) does not fit on GPU " +
                  std::to_string(P.device);
            return false;
        }
        cudaMemset(P.arena, 0, count.used);
        Bump real;
        real.base = (uint8_t*) P.arena;
        carve(P, real);
        total += count.used;
        if (cudaStreamCreateWithFlags(&P.copy, cudaStreamNonBlocking) != cudaSuccess ||
            cudaStreamCreateWithFlags(&P.cs, cudaStreamNonBlocking) != cudaSuccess ||
            cudaStreamCreateWithFlags(&P.side, cudaStreamNonBlocking) != cudaSuccess ||
            cudaEventCreateWithFlags(&P.fork, cudaEventDisableTiming) != cudaSuccess ||
            cudaEventCreateWithFlags(&P.join, cudaEventDisableTiming) != cudaSuccess) {
            err = "verify: stream create failed";
            return false;
        }
    }
    sink_.staging = (unsigned long long) stages_[0].staging_;
    sink_.staging_cap = kStagingBlobs;
    std::fprintf(stderr, "strata verify: window up to %d tokens, %.1f MiB of device buffers\n", max_t,
                 (double) total / 1048576.0);
    return true;
}

const float* Verifier::final_R(int t) const {
    return stages_.back().R_ + (size_t) t * (size_t) (g_->hc * g_->n_embd);
}

bool Verifier::head_logits(int T, float* out, std::string& err) const {
    const Part& L = stages_.back();
    DeviceGuard dg(L.device);
    if (cudaStreamSynchronize(L.cs) != cudaSuccess ||
        cudaMemcpy(out, L.head_logits_, (size_t) T * (size_t) n_vocab_ * sizeof(float), cudaMemcpyDeviceToHost) !=
            cudaSuccess) {
        err = "verify: reading the head logits failed";
        return false;
    }
    return true;
}

Verifier::Part& Verifier::part_of(int64_t layer) {
    for (Part& P : stages_)
        if (layer < P.l1) return P;
    return stages_.back();
}

// ================================ THE WINDOW, AS CAPTURED ================================
//
// Plan v0.3 P6 (split window): with `groups_ == 2` the window's tokens are cut into two groups A = [0, T/2 up) and
// B = the rest, and the stream is ordered
//
//     pre(0,A) pre(0,B) | post(0,A) pre(1,A) | post(0,B) pre(1,B) | post(1,A) pre(2,A) | ...
//
// so the CPU computes A's experts of layer l while the GPU runs B's mixer and router of layer l, and B's experts
// while the GPU combines A and runs A's layer l+1.  B's mixer only needs A's mixer of the same layer (K/V, GDN
// state), never A's experts, so nothing waits that did not wait before.  Every token's arithmetic is unchanged.
//
// Pipeline stages (placement.hpp): each stage's graph runs its own layers [l0, l1).  A stage after the first starts
// by waiting ON ITS GPU for the previous stage's hand-off flag and copying the residual it published to mapped host
// memory; a stage before the last ends by applying its last layer's pending write (the next read that would fold it
// runs on another GPU) and publishing the residual.  The last stage runs the head.
bool Verifier::record_window(int T, Part& P, std::string& err) {
    using namespace strata::kernels;
    const cudaStream_t cs = P.cs;
    const ModelGeometry& g = *g_;
    const WeightTable& wt = *wt_;
    SessionState& ss = *ss_;
    SessionStage& pst = ss.stages[(size_t) P.stage];
    const bool first = P.stage == 0, last = P.stage == (int) stages_.size() - 1;
    const int64_t N = g.n_embd, HC = g.hc, K = ss.k, C = g.ssm_conv_channels, ZV = g.ssm_value_dim;
    const int64_t HV = g.ssm_v_heads, HK = g.ssm_k_heads, NH = g.n_head, HD = g.head_dim, NKV = g.n_head_kv;
    const int64_t IQ = g.idx_q_heads, ID = g.idx_key_dim, NE = g.n_expert, MT = max_t_;
    const QsaShapes s = shapes_of(g);
    const GrShapes gs{g.n_embd, g.hc, g.hc_lr};
    const int64_t HS = (int64_t) NG_HIST * NG_HC_DIM;
    const int64_t TS = (s.idx_block - 1) * ID;
    const bool ple_on = ss.ple.ready() && P.l0 <= 1 && 1 < P.l1;   // the PLE runs at layer 1, on its stage
    auto Rt = [&](int t) { return P.R_ + (size_t) t * HC * N; };
    const int G = (split_ && T >= 2) ? 2 : 1;
    const int tb_[2] = {0, (T + 1) / 2}, te_[2] = {G == 2 ? (T + 1) / 2 : T, T};
    groups_[T] = G;

    // ---- the window's inputs, from mapped staging
    if (first) copy_i32_from_mapped(P.tok_, m_tok_, T, cs);
    copy_i32_from_mapped(P.step_, m_step_, (int64_t) T * kStepCount, cs);
    copy_i32_from_mapped(P.pos_, m_pos_, (int64_t) T * NH, cs);
    copy_i32_from_mapped(P.pos_kv_, m_pos_kv_, (int64_t) T * NKV, cs);
    copy_i32_from_mapped(P.pos_iq_, m_pos_iq_, (int64_t) T * IQ, cs);
    if (ple_on) copy_from_mapped(P.ple_, m_ple_, (int64_t) T * N, cs);

    // ---- the embeddings, broadcast to the hc streams; or the residual the previous stage handed on
    if (!first) {
        wait_flag_ge(m_hand_flag_[(size_t) P.stage - 1], 1u, cs);
        copy_from_mapped(P.R_, m_hand_[(size_t) P.stage - 1], (int64_t) T * HC * N, cs);
    } else if (const NativeEmbed* ne = native_embed()) {       // plan v0.3 P6: the GGUF-form table
        ne->gather_dev(P.tok_, T, P.emb_, cs);
        broadcast_streams(P.emb_, P.R_, N, (int) HC, T, cs);
    } else {
        const WeightRef* w = wt.find("token_embd.weight");
        if (w == nullptr || w->codebook_iq4nl || (w->code_bits != 2 && w->code_bits != 4 && w->code_bits != 8)) {
            err = "verify: token_embd.weight is missing or not an S2/S4/S8 tensor";
            return false;
        }
        const auto* codes = (const uint8_t*) w->data;
        const auto* scales = (const float*) (codes + w->codes_bytes);
        const auto* offsets = w->has_offset ? (const float*) (codes + w->codes_bytes + w->scales_bytes) : nullptr;
        const uint64_t row_codes = (uint64_t) (w->ne0 / (8 / w->code_bits));
        const uint64_t row_groups = (uint64_t) (w->ne0 / w->group_elems);
        embedding_gather_dev(codes, scales, offsets, P.tok_, T, w->ne0, w->code_bits, w->code_bias, w->group_elems,
                             row_codes, row_groups, P.emb_, cs);
        broadcast_streams(P.emb_, P.R_, N, (int) HC, T, cs);
    }

    // per-layer state indices (GDN and QSA layers are numbered separately)
    std::vector<int64_t> gdn_idx((size_t) g.n_layers, -1), qsa_idx((size_t) g.n_layers, -1);
    {
        int64_t qi = 0, gi = 0;
        for (int64_t l = 0; l < g.n_layers; ++l) {
            if (is_qsa_layer(g, l)) qsa_idx[(size_t) l] = qi++;
            else gdn_idx[(size_t) l] = gi++;
        }
    }

    // ---------------------------------------------------------------- pre(l, group): up to the ring
    auto pre = [&](int64_t l, int grp) -> bool {
        const int tb = tb_[grp], te = te_[grp], n = te - tb;
        const LayerView v(wt, l);
        const char* pfx[2] = {"hc_attn_", "hc_ffn_"};
        const WeightRef *wn[2], *wd[2], *wu[2], *wi[2];
        for (int h = 0; h < 2; ++h) {
            wn[h] = need(v, (std::string(pfx[h]) + "norm.weight").c_str(), err);
            wd[h] = need(v, (std::string(pfx[h]) + "down.weight").c_str(), err);
            wu[h] = need(v, (std::string(pfx[h]) + "up.weight").c_str(), err);
            wi[h] = need(v, (std::string(pfx[h]) + "inject.weight").c_str(), err);
            if (!wn[h] || !wd[h] || !wu[h] || !wi[h]) return false;
        }
        // the previous layer's FFN write, folded into this layer's first read (a control vector after it has
        // already applied it, and the previous stage when the layer is this stage's first)
        bool pending = l > P.l0 && !cvec().covers(l - 1);
        if (l == 1 && ple_on) {
            float* normalized = (float*) ((uint8_t*) ss.ple.scratch + ple_block_scratch_bytes());
            // the key and value projections depend on the token's own embedding only: one pass for the group
            const bool rows = ple_project_rows_available(ss.ple.w);
            if (rows) {
                try {
                    ple_project_rows(P.ple_ + tb * N, n, ss.ple.w, P.ple_key_ + (size_t) tb * NG_HC_DIM,
                                     P.ple_val_ + tb * N, cs);
                } catch (const std::exception& e) {
                    err = std::string("verify PLE: ") + e.what();
                    return false;
                }
            }
            for (int t = tb; t < te; ++t) {
                if (pending) gr_write(Rt(t), P.bo_ + t * N, P.inj2_ + t * HC, gs, Rt(t), cs);
                PleOut po;
                po.normalized = normalized;
                po.result = Rt(t);
                try {
                    ple_block(P.ple_ + t * N, Rt(t), ss.ple.hist, ss.ple.w, po, ss.ple.scratch, cs,
                              rows ? P.ple_key_ + (size_t) t * NG_HC_DIM : nullptr, rows ? P.ple_val_ + t * N : nullptr);
                    ple_history_advance(ss.ple.hist, normalized, cs);
                } catch (const std::exception& e) {
                    err = std::string("verify PLE: ") + e.what();
                    return false;
                }
                copy_from_mapped(P.hist_snap_ + (size_t) t * HS, ss.ple.hist, HS, cs);
            }
            pending = false;
        }
        auto gr_read_group = [&](int half, bool apply, float* inj_prev, float* inj_out) {
            FusedGrArgs fa[kFusedGrMaxT];
            for (int t = tb; t < te; ++t) {
                FusedGrArgs& a = fa[t - tb];
                a.R = Rt(t); a.R_out = Rt(t); a.apply = apply;
                a.bo_prev = P.bo_ + t * N; a.inj_prev = inj_prev + t * HC;
                a.w_norm = (const float*) wn[half]->data; a.w_down = (const uint16_t*) wd[half]->data;
                a.w_up = (const uint16_t*) wu[half]->data; a.w_inject = (const uint16_t*) wi[half]->data;
                a.eps = EPS; a.lo = P.lo_ + t * g.hc_lr; a.rs = P.rs_ + t * HC;
                a.inject_out = inj_out + t * HC; a.mixed = P.mixed_ + t * N;
            }
            fused_gr_read_multi(fa, n, P.xn_ + (size_t) tb * HC * N, cs);
        };
        gr_read_group(0, pending, P.inj2_, P.inj_);
        float* xm = P.mixed_ + tb * N;
        try {
            if (!is_qsa_layer(g, l)) {
                // ======================= GDN =======================
                const WeightRef *wqkv = need(v, "attn_qkv.weight", err), *wg = need(v, "attn_gate.weight", err),
                                *wout = need(v, "ssm_out.weight", err), *wa = need(v, "ssm_alpha.weight", err),
                                *wb = need(v, "ssm_beta.weight", err), *wc = need(v, "ssm_conv1d.weight", err),
                                *wnm = need(v, "ssm_norm.weight", err), *wdt = need(v, "ssm_dt.bias", err),
                                *wsa = need(v, "ssm_a", err);
                if (!wqkv || !wg || !wout || !wa || !wb || !wc || !wnm || !wdt || !wsa) return false;
                if (!native_of(wqkv, v.name("attn_qkv.weight"), err) || !native_of(wg, v.name("attn_gate.weight"), err) ||
                    !native_of(wout, v.name("ssm_out.weight"), err))
                    return false;
                const int64_t gi = gdn_idx[(size_t) l], gl = gi - P.gdn0;
                float* state = ss.gdn_states[(size_t) gi];
                float* conv = state + (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size;
                float* qkv = P.qkv_L_ + (size_t) gl * MT * C;
                float* hb = P.h_L_ + (size_t) gl * MT * C;
                float* gate = P.gate_L_ + (size_t) gl * MT * HV;
                float* beta = P.beta_L_ + (size_t) gl * MT * HV;
                native_quantize_q8_1(xm, P.xq_, (int) N, n, cs);
                native_mmvq(wqkv->native_type, wqkv->native_data, P.xq_, qkv + (size_t) tb * C, (int) N, (int) C, n, cs);
                gdn_conv_l2_multi(conv, qkv, (const float*) wc->data, hb, (int) C, (int) (2 * HK), EPS, n, cs, tb);
                gdn_ab_multi(xm, (const uint16_t*) wa->data, (const uint16_t*) wb->data, (const float*) wdt->data,
                             (const float*) wsa->data, gate + (size_t) tb * HV, beta + (size_t) tb * HV, (int) N, (int) HV,
                             n, cs);
                native_mmvq(wg->native_type, wg->native_data, P.xq_, P.z_ + (size_t) tb * ZV, (int) N, (int) ZV, n, cs);
                // the recurrence from the untouched state over tokens [0, te); outputs only for this group's
                gdn_step_norm_multi(state, hb, (int) C, gate, beta, P.z_, (const float*) wnm->data, EPS, P.y_, (int) HK,
                                    (int) HV, te, nullptr, cs, tb);
                native_quantize_q8_1(P.y_ + (size_t) tb * ZV, P.xq_, (int) ZV, n, cs);
                native_mmvq(wout->native_type, wout->native_data, P.xq_, P.bo_ + tb * N, (int) ZV, (int) N, n, cs);
            } else {
                // ======================= QSA =======================
                const int64_t qi = qsa_idx[(size_t) l];
                const QsaState& st = ss.qsa_states[qi];
                const WeightRef *wik = need(v, "indexer.k_proj.weight", err), *wq = need(v, "attn_q.weight", err),
                                *wk = need(v, "attn_k.weight", err), *wv = need(v, "attn_v.weight", err),
                                *wo = need(v, "attn_output.weight", err), *wiq = need(v, "indexer.q_proj.weight", err),
                                *wqn = need(v, "attn_q_norm.weight", err), *wkn = need(v, "attn_k_norm.weight", err),
                                *wiqn = need(v, "indexer.q_norm.weight", err), *wikn = need(v, "indexer.k_norm.weight", err);
                if (!wik || !wq || !wk || !wv || !wo || !wiq || !wqn || !wkn || !wiqn || !wikn) return false;
                if (!native_of(wq, v.name("attn_q.weight"), err) || !native_of(wk, v.name("attn_k.weight"), err) ||
                    !native_of(wv, v.name("attn_v.weight"), err) || !native_of(wo, v.name("attn_output.weight"), err))
                    return false;
                auto norm_rope = [&](float* data, const WeightRef* norm, int rows, int cols, const int32_t* pos) {
                    if (native_qsa_enabled()) native_qsa_rms_norm_weighted(data, (const float*) norm->data, data, cols, rows, EPS, cs);
                    else rms_norm_weighted(data, (const float*) norm->data, rows, cols, EPS, cs);
                    if (native_rope_enabled()) native_rope_apply(data, data, rows, cols, (int) s.n_rot, (float) qsa_freq_base(), pos, cs);
                    else rope_neox_apply(data, data, rows, cols, (int) s.n_rot, st.cos_tab, st.sin_tab, pos, cs);
                };
                float* idx_raw = P.idx_raw_L_ + (size_t) (qi - P.qsa0) * MT * ID;
                native_quantize_q8_1(xm, P.xq_, (int) N, n, cs);
                bf16_gemv_fp32_mmvf_rows(P.mixed_ + tb * N, N, (const uint16_t*) wik->data, idx_raw + tb * ID, ID, N, ID, n, cs);
                native_mmvq(wk->native_type, wk->native_data, P.xq_, P.kcur_ + tb * NKV * HD, (int) N, (int) (NKV * HD), n, cs);
                native_mmvq(wv->native_type, wv->native_data, P.xq_, P.vcur_ + tb * NKV * HD, (int) N, (int) (NKV * HD), n, cs);
                norm_rope(P.kcur_ + tb * NKV * HD, wkn, (int) (n * NKV), (int) HD, P.pos_kv_ + tb * NKV);
                if (st.kv_q4) {   // Q4_0 KV (kv_q4.hpp): K and V rotated before they are stored
                    fwht256_inplace_cuda(P.kcur_ + tb * NKV * HD, (int64_t) n * NKV, cs);
                    fwht256_inplace_cuda(P.vcur_ + tb * NKV * HD, (int64_t) n * NKV, cs);
                }
                if (grp == 0) copy_from_mapped(P.tail_snap_ + (size_t) (qi - P.qsa0) * TS, st.idx_tail, TS, cs);
                for (int t = tb; t < te; ++t) {
                    const int32_t* step_t = P.step_ + t * kStepCount;
                    if (st.kv_q4)
                        kv_append_q4_step(st.k_q4, st.v_q4, st.page_table, step_t, P.kcur_ + t * NKV * HD,
                                          P.vcur_ + t * NKV * HD, s, cs, &st.host);
                    else if (st.kv_int8)
                        kv_append_q8_step(st.k_q, st.v_q, st.k_scale, st.v_scale, st.page_table, step_t,
                                          P.kcur_ + t * NKV * HD, P.vcur_ + t * NKV * HD, s, cs, &st.host);
                    else
                        kv_append_step(st.k_pool, st.v_pool, st.page_table, step_t, P.kcur_ + t * NKV * HD,
                                       P.vcur_ + t * NKV * HD, s, cs, &st.host);
                }
                const QsaIndexerBuffers ib{st.idx_tail, st.idx_dead, st.idx_pooled, st.idx_block_pos};
                for (int t = tb; t < te; ++t)
                    native_qsa_indexer_append(idx_raw + t * ID, P.step_ + t * kStepCount + kStepPos, 0,
                                              (const float*) wikn->data, EPS, ib, s, st.max_cells,
                                              (float) qsa_freq_base(), cs);
                native_mmvq(wq->native_type, wq->native_data, P.xq_, P.qfull_ + tb * NH * 2 * HD, (int) N, (int) (NH * 2 * HD),
                            n, cs);
                {   // the group's q halves: one strided copy, then every head row normed and rotated at once
                    float* qc = P.qcur_ + tb * NH * HD;
                    if (cudaMemcpy2DAsync(qc, (size_t) HD * 4, P.qfull_ + tb * NH * 2 * HD, (size_t) HD * 2 * 4,
                                          (size_t) HD * 4, (size_t) (n * NH), cudaMemcpyDeviceToDevice, cs) != cudaSuccess) {
                        err = "verify: the q/gate split failed";
                        return false;
                    }
                    norm_rope(qc, wqn, (int) (n * NH), (int) HD, P.pos_ + tb * NH);
                    if (st.kv_q4) fwht256_inplace_cuda(qc, n * NH, cs);   // <Hq, Hk> = <q, k>
                }
                bf16_gemv_fp32_mmvf_rows(P.mixed_ + tb * N, N, (const uint16_t*) wiq->data, P.qidx_ + tb * IQ * ID, IQ * ID, N,
                                         IQ * ID, n, cs);
                norm_rope(P.qidx_ + tb * IQ * ID, wiqn, (int) (n * IQ), (int) ID, P.pos_iq_ + tb * IQ);
                qsa_block_scores(st.idx_pooled, st.idx_dead, P.qidx_ + tb * IQ * ID, P.step_ + tb * kStepCount, n, max_blocks_,
                                 s, P.scores_ + (size_t) tb * max_blocks_, cs);
                qsa_block_topk(P.scores_ + (size_t) tb * max_blocks_, P.step_ + tb * kStepCount, n, max_blocks_, cap_, s,
                               P.sel_ + (size_t) tb * cap_, cs);
                // KV streaming: the n selections' blocks resident (device-side, inside the graph)
                qsa_kv_resolve(st, *g_, P.sel_ + (size_t) tb * cap_, P.step_ + tb * kStepCount, n, cap_, cs);
                const QsaAttnPools pools = qsa_attn_pools(st);
                qsa_decode_attn_batch(P.qcur_ + tb * NH * HD, pools, P.sel_ + (size_t) tb * cap_, P.step_ + tb * kStepCount, cap_,
                                      s, P.attn_scratch_ + (size_t) tb * attn_scratch_floats_, P.attn_ + tb * NH * HD, n, cs);
                if (st.kv_q4) fwht256_inplace_cuda(P.attn_ + tb * NH * HD, (int64_t) n * NH, cs);   // back: H^-1 = H
                if (native_qsa_enabled())
                    native_qsa_gate_apply(P.attn_ + tb * NH * HD, P.qfull_ + tb * NH * 2 * HD, P.attn32_ + tb * NH * HD,
                                          (int) (n * NH), (int) HD, cs);
                else
                    for (int t = tb; t < te; ++t)
                        qsa_gate_apply_f32(P.attn_ + t * NH * HD, P.qfull_ + t * NH * 2 * HD, s, P.attn32_ + t * NH * HD, cs);
                native_quantize_q8_1(P.attn32_ + tb * NH * HD, P.xq_, (int) (NH * HD), n, cs);
                native_mmvq(wo->native_type, wo->native_data, P.xq_, P.bo_ + tb * N, (int) (NH * HD), (int) N, n, cs);
            }
        } catch (const std::exception& e) {
            err = "verify layer " + std::to_string(l) + ": " + e.what();
            return false;
        }
        gr_read_group(1, true, P.inj_, P.inj2_);
        {
            MoEBuffers mb = pst.moe;
            mb.logits = P.logits_ + tb * NE; mb.ids = P.ids_ + tb * K; mb.weights = P.w_ + tb * K;
            if (!moe_route_rows(wt, g, l, K, mb, P.mixed_ + tb * N, n, NE, cs, err)) return false;
        }
        doorbell_publish(xm, P.ids_ + tb * K, P.w_ + tb * K, (int64_t) n * N, (int64_t) n * K, m_x_ + tb * N,
                         m_ids_ + tb * K, m_w_ + tb * K, m_seq_, cs);
        {
            const WeightRef *wgi = need(v, "ffn_gate_inp_shexp.weight", err), *wsg = need(v, "ffn_gate_shexp.weight", err),
                            *wsu = need(v, "ffn_up_shexp.weight", err), *wsd = need(v, "ffn_down_shexp.weight", err);
            if (!wgi || !wsg || !wsu || !wsd) return false;
            if (!native_of(wsg, v.name("ffn_gate_shexp.weight"), err) || !native_of(wsu, v.name("ffn_up_shexp.weight"), err) ||
                !native_of(wsd, v.name("ffn_down_shexp.weight"), err))
                return false;
            NativeSharedWeights nsw;
            nsw.gate_type = wsg->native_type; nsw.gate_data = wsg->native_data;
            nsw.up_type = wsu->native_type; nsw.up_data = wsu->native_data;
            nsw.down_type = wsd->native_type; nsw.down_data = wsd->native_data;
            nsw.q8_1 = P.xq_;
            if (!shared_expert_native_bf16())   // the scalar gate's BF16 input (the native gate reads the floats)
                f32_to_bf16_bulk(P.mixed_ + tb * N, P.sh_bf16_ + tb * N, (int64_t) n * N, cs);
            try {
                shared_expert_multi(n, xm, P.sh_bf16_ + tb * N, nsw, (const uint16_t*) wgi->data, P.sh_gate_ + (size_t) tb * g.n_ff,
                                    P.sh_up_ + (size_t) tb * g.n_ff, P.sh_g_ + tb, P.shared_ + tb * N, N, g.n_ff, cs);
            } catch (const std::exception& e) {
                err = std::string("verify shared expert: ") + e.what();
                return false;
            }
        }
        if (strata::kernels::cpu::expert_layout().native)
            quantize_q8_1_rows(xm, n, N, P.nat_xq_ + (size_t) tb * (N / 32) * 36, cs);
        else
            quantize_q8_0_scaled(xm, P.hit_xq_ + (size_t) tb * (N / 32) * 34, P.hit_xs_ + (size_t) tb * (N / 32), (int64_t) n * N, cs);
        return true;
    };

    // ---------------------------------------------------------------- post(l, group): experts, combine
    auto post = [&](int64_t l, int grp) -> bool {
        const int tb = tb_[grp], te = te_[grp], n = te - tb;
        const uint32_t ring = (uint32_t) (l * G + grp + 1);
        wait_flag_ge(m_flagA_, ring, cs);                      // the pool published this group's GPU plan
        const int64_t cap = (int64_t) n * K, capx = (int64_t) max_t_ * K;
        int32_t* pl = P.plan_ + (size_t) grp * (size_t) (plan_i32_ + 16);
        copy_i32_from_mapped(pl, m_plan_ + (size_t) grp * (size_t) plan_i32_, plan_i32_, cs);
        const int32_t* p_counts = pl;
        const int32_t* p_start = pl + 4;
        const int32_t* p_dst = p_start + capx + 1;
        const int32_t* p_tok = p_dst + capx;
        const int64_t ptr_off = ((4 + (capx + 1) + 2 * capx) + 1) & ~1ll;
        const unsigned long long* p_ptr = (const unsigned long long*) (pl + ptr_off);
        const unsigned long long* p_ptr2 = p_ptr + capx;
        const int32_t* p_start2 = pl + ptr_off + 4 * capx;
        float* hit_out = P.hit_out_ + (size_t) tb * K * N;
        const auto& lay = strata::kernels::cpu::expert_layout();
        // plan v0.3 P6: the VRAM groups now; the PCIe groups once the copy engine has landed them in staging
        auto grouped = [&](const unsigned long long* gp, const int32_t* gs, const int32_t* gn) {
            if (lay.native) {
                // the layer's GGUF formats (i-quant gate/up, Q2_0 / IQ4_NL down)
                const auto& f = lay.fmt[(size_t) l];
                const NativeExpertLayout L = native_expert_layout(f.gu_type, f.d_type, f.n_embd, f.n_ff);
                native_expert_grouped(L, gp, gs, gn, p_dst, p_tok, cap, cap,
                                      P.nat_xq_ + (size_t) tb * (N / 32) * 36, P.hit_scratch_, hit_out, cs);
            } else {
                moe_grouped_s2(gp, gs, gn, p_dst, p_tok, cap, cap, P.hit_xq_ + (size_t) tb * (N / 32) * 34,
                               P.hit_xs_ + (size_t) tb * (N / 32), P.hit_scratch_, hit_out, cs);
            }
        };
        if (sink_.pcie_mode == 2) {
            // the PCIe share: a copy kernel stages it on the side stream while the VRAM experts run, then the pointers
            // are rebased to staging
            const int64_t per = G == 2 ? kStagingBlobs / 2 : kStagingBlobs;
            uint8_t* stage = P.staging_ + (size_t) (grp * per) * lay.max_blob;
            cudaEventRecord(P.fork, cs);
            cudaStreamWaitEvent(P.side, P.fork, 0);
            wait_flag_ge(m_flagB_, ring, P.side);
            fetch_blobs(p_ptr2, p_counts + 2, stage, (int64_t) lay.blob_bytes(l), (int) per, P.side);
            rebase_ptrs((unsigned long long*) p_ptr2, p_counts + 2, stage, (int64_t) lay.blob_bytes(l), P.side);
            cudaEventRecord(P.join, P.side);
            grouped(p_ptr, p_start, p_counts);
            cudaStreamWaitEvent(cs, P.join, 0);
        } else {
            grouped(p_ptr, p_start, p_counts);
            wait_flag_ge(m_flagB_, ring, cs);                  // the PCIe share is in staging (DMA) or mapped
        }
        grouped(p_ptr2, p_start2, p_counts + 2);
        wait_flag_ge(m_flag_, ring, cs);                       // the CPU's share is in the mapped rows
        copy_rows_from_mapped(P.parts_ + (size_t) tb * K * N, m_ymiss_ + (size_t) tb * K * N, p_dst, p_counts + 1,
                              (int64_t) n * K, N, cs);
        moe_hit_add(P.parts_ + (size_t) tb * K * N, hit_out, p_dst, p_counts + 1, cap, N, cs);
        {
            MoEBuffers mb = pst.moe;
            mb.weights = P.w_ + tb * K; mb.shared = P.shared_ + tb * N;
            if (!moe_combine_parts_rows(g, l, K, mb, P.parts_ + (size_t) tb * K * N, P.bo_ + tb * N, n, cs, err)) return false;
        }
        if (l == g.n_layers - 1) {
            for (int t = tb; t < te; ++t) gr_write(Rt(t), P.bo_ + t * N, P.inj2_ + t * HC, gs, Rt(t), cs);
            if (cvec().covers(l)) cvec_apply(Rt(tb), l, n, HC * N, nullptr, 0, nullptr, 0, false, cs);
        } else if (cvec().covers(l)) {
            cvec_apply(Rt(tb), l, n, HC * N, P.bo_ + tb * N, N, P.inj2_ + tb * HC, HC, true, cs);
        } else if (l == P.l1 - 1) {
            // the stage's last layer: its write is applied as the next layer's first step would apply it (the PLE
            // layer's plain write, else the fused read's)
            if (l == 0 && ss.ple.ready())
                for (int t = tb; t < te; ++t) gr_write(Rt(t), P.bo_ + t * N, P.inj2_ + t * HC, gs, Rt(t), cs);
            else
                fused_gr_write(Rt(tb), P.bo_ + tb * N, P.inj2_ + tb * HC, n, cs);
        }
        return true;
    };

    for (int grp = 0; grp < G; ++grp)
        if (!pre(P.l0, grp)) return false;
    for (int64_t l = P.l0; l < P.l1; ++l)
        for (int grp = 0; grp < G; ++grp) {
            if (!post(l, grp)) return false;
            if (l + 1 < P.l1 && !pre(l + 1, grp)) return false;
        }
    if (!last) {
        // the residual to the next stage: published to mapped memory, then its flag (ordered by the kernel's fence)
        doorbell_publish(P.R_, nullptr, nullptr, (int64_t) T * HC * N, 0, m_hand_[(size_t) P.stage], nullptr, nullptr,
                         m_hand_flag_[(size_t) P.stage], cs);
        if (!(ep_ && first)) return true;
        // expert parallelism: the later layers' entries whose experts are in this GPU's VRAM.  A layer the pool gave
        // none of them finds another layer's tag in the block and computes nothing.
        const auto& lay = strata::kernels::cpu::expert_layout();
        for (int64_t l = sink_.ep_layer0; l < g.n_layers; ++l)
            for (int grp = 0; grp < G; ++grp) {
                const int tb = tb_[grp], n = te_[grp] - tb;
                const uint32_t ring = (uint32_t) (l * G + grp + 1);
                const int64_t cap = (int64_t) n * K, capx = (int64_t) max_t_ * K;
                int32_t* pl = P.plan_ + (size_t) grp * (size_t) (plan_i32_ + 16);
                wait_flag_ge(m_epA_, ring, cs);
                take_plan(pl, m_eplan_ + (size_t) grp * (size_t) plan_i32_, plan_i32_, (int32_t) ring, cs);
                const int32_t* p_start = pl + 4;
                const int32_t* p_dst = p_start + capx + 1;
                const int32_t* p_tok = p_dst + capx;
                const unsigned long long* p_ptr = (const unsigned long long*) (pl + (((4 + (capx + 1) + 2 * capx) + 1) & ~1ll));
                copy_from_mapped_if(P.bo_ + tb * N, m_x_ + tb * N, (int64_t) n * N, pl, cs);
                quantize_q8_1_rows(P.bo_ + tb * N, n, N, P.nat_xq_ + (size_t) tb * (N / 32) * 36, cs);
                const auto& f = lay.fmt[(size_t) l];
                native_expert_grouped(native_expert_layout(f.gu_type, f.d_type, f.n_embd, f.n_ff), p_ptr, p_start, pl, p_dst,
                                      p_tok, cap, cap, P.nat_xq_ + (size_t) tb * (N / 32) * 36, P.hit_scratch_,
                                      P.hit_out_ + (size_t) tb * K * N, cs);
                rows_to_mapped(m_ymiss_ + (size_t) tb * K * N, P.hit_out_ + (size_t) tb * K * N, p_dst, pl + 1, cap, N, cs);
                set_flag(m_epB_, ring, cs);
            }
        return true;
    }

    // ---- the head, T columns, and the argmax of each
    {
        const WeightRef *hn = wt.find("output_hc_norm.weight"), *hd = wt.find("output_hc_down.weight"),
                        *hu = wt.find("output_hc_up.weight");
        if (!hn || !hd || !hu) { err = "verify: an output_hc_* weight is missing"; return false; }
        for (int t = 0; t < T; ++t) {
            BlockBuffers bb = pst.block;
            bb.R = Rt(t);
            bb.mixed = P.head_mixed_ + t * N;
            if (head_ != nullptr && head_->loaded()) {
                if (!lm_head_mix(wt, g, bb, cs, err)) return false;
            } else if (!lm_head(wt, g, bb, P.head_logits_ + (size_t) t * n_vocab_, cs, err)) {
                return false;
            }
        }
        if (head_ != nullptr && head_->loaded()) {
            try {
                native_quantize_q8_1(P.head_mixed_, P.xq_, (int) N, T, cs);
                native_mmvq(head_->type(), head_->weights(), P.xq_, P.head_logits_, (int) N, (int) n_vocab_, T, cs);
            } catch (const std::exception& e) {
                err = std::string("verify head: ") + e.what();
                return false;
            }
        }
        // Greedy, the default, is recorded here as before (no extra launch or sync per window). A request that
        // samples or penalizes is sampled again host-side after the replay (run()) with its own parameters and a
        // fresh draw counter: a captured sampler would bake them in and replay the same draws forever.
        SamplerParams sp;
        sp.greedy = true;
        sp.temperature = 0.0f;
        sample_tokens(P.head_logits_, T, (int) n_vocab_, nullptr, 0, sp, m_out_, cs);
    }
    return true;
}

bool Verifier::capture(int T, std::string& err) {
    if (stages_[0].exec[T] != nullptr) return true;
    for (Part& P : stages_) {
        DeviceGuard dg(P.device);
        if (cudaStreamBeginCapture(P.cs, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
            err = "verify: begin capture failed";
            return false;
        }
        std::string rerr;
        const bool ok = record_window(T, P, rerr);
        cudaGraph_t graph = nullptr;
        const cudaError_t ce = cudaStreamEndCapture(P.cs, &graph);
        if (!ok) {
            if (graph) cudaGraphDestroy(graph);
            err = rerr;
            return false;
        }
        if (ce != cudaSuccess) {
            err = std::string("verify: end capture: ") + cudaGetErrorString(ce);
            return false;
        }
        const cudaError_t ie = cudaGraphInstantiate(&P.exec[T], graph, 0);
        cudaGraphDestroy(graph);
        if (ie != cudaSuccess) {
            err = std::string("verify: instantiate: ") + cudaGetErrorString(ie);
            return false;
        }
        const cudaError_t ue = cudaGraphUpload(P.exec[T], P.cs);
        const cudaError_t us = cudaStreamSynchronize(P.cs);
        std::fprintf(stderr, "strata verify: captured the %d-token window on GPU %d (layers %lld-%lld; upload %s, sync %s)\n",
                     T, P.device, (long long) P.l0, (long long) (P.l1 - 1), cudaGetErrorString(ue),
                     cudaGetErrorString(us));
    }
    return true;
}

bool Verifier::record_commit(Part& P, std::string& err) {
    using namespace strata::kernels;
    const cudaStream_t cs = P.cs;
    const ModelGeometry& g = *g_;
    SessionState& ss = *ss_;
    const QsaShapes s = shapes_of(g);
    const int64_t C = g.ssm_conv_channels, HV = g.ssm_v_heads, ID = g.idx_key_dim, MT = max_t_;
    const int64_t TS = (s.idx_block - 1) * ID;
    const int64_t HS = (int64_t) NG_HIST * NG_HC_DIM;
    try {
        copy_i32_from_mapped(P.commit_, m_commit_, 2 + MT, cs);
        int64_t qsa_index = P.qsa0, gdn_index = P.gdn0;
        for (int64_t l = P.l0; l < P.l1; ++l) {
            const LayerView v(*wt_, l);
            if (!is_qsa_layer(g, l)) {
                const WeightRef* wnm = need(v, "ssm_norm.weight", err);
                if (!wnm) return false;
                const int64_t gl = gdn_index - P.gdn0;
                float* state = ss.gdn_states[(size_t) gdn_index];
                float* conv = state + (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size;
                const float* qkv = P.qkv_L_ + (size_t) gl * MT * C;
                gdn_conv_commit(conv, qkv, (int) C, P.commit_, cs);
                gdn_step_norm_multi(state, P.h_L_ + (size_t) gl * MT * C, (int) C, P.gate_L_ + (size_t) gl * MT * HV,
                                    P.beta_L_ + (size_t) gl * MT * HV, P.z_, (const float*) wnm->data, EPS, P.y_dummy_,
                                    (int) g.ssm_k_heads, (int) HV, (int) MT, P.commit_, cs);
                ++gdn_index;
            } else {
                const QsaState& st = ss.qsa_states[qsa_index];
                const WeightRef* wikn = need(v, "indexer.k_norm.weight", err);
                if (!wikn) return false;
                const int64_t ql = qsa_index - P.qsa0;
                copy_from_mapped(st.idx_tail, P.tail_snap_ + (size_t) ql * TS, TS, cs);
                const QsaIndexerBuffers ib{st.idx_tail, st.idx_dead, st.idx_pooled, st.idx_block_pos};
                for (int64_t t = 0; t < MT; ++t)
                    native_qsa_indexer_append(P.idx_raw_L_ + (size_t) (ql * MT + t) * ID, P.commit_ + 2 + t, 0,
                                              (const float*) wikn->data, EPS, ib, s, st.max_cells,
                                              (float) qsa_freq_base(), cs);
                ++qsa_index;
            }
        }
        if (ss.ple.ready() && P.l0 <= 1 && 1 < P.l1) copy_indexed(ss.ple.hist, P.hist_snap_, HS, P.commit_ + 1, HS, cs);
    } catch (const std::exception& e) {
        err = std::string("verify commit: ") + e.what();
        return false;
    }
    return true;
}

bool Verifier::capture_commit(std::string& err) {
    if (stages_[0].commit_exec != nullptr) return true;
    for (Part& P : stages_) {
        DeviceGuard dg(P.device);
        if (cudaStreamBeginCapture(P.cs, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
            err = "verify: begin commit capture failed";
            return false;
        }
        const bool ok = record_commit(P, err);
        cudaGraph_t graph = nullptr;
        const cudaError_t ce = cudaStreamEndCapture(P.cs, &graph);
        if (!ok) {
            if (graph) cudaGraphDestroy(graph);
            return false;
        }
        if (ce != cudaSuccess || cudaGraphInstantiate(&P.commit_exec, graph, 0) != cudaSuccess) {
            if (graph) cudaGraphDestroy(graph);
            err = std::string("verify: commit capture: ") + cudaGetErrorString(ce);
            return false;
        }
        cudaGraphDestroy(graph);
    }
    return true;
}

bool Verifier::run(int T, const int32_t* tokens, int64_t pos0, PoolMultiFn pool, void* user, int32_t* out,
                   std::string& err) {
    using namespace strata::kernels;
    if (T < 1 || T > max_t_) { err = "verify: window size out of range"; return false; }
    const ModelGeometry& g = *g_;
    SessionState& ss = *ss_;
    if (pos0 + T > ss.qsa_states[0].max_cells) { err = "verify: the window runs past the context"; return false; }
    if (!capture(T, err) || !capture_commit(err)) return false;
    VDBG("captured; staging\n");
    const Clock::time_point t0 = Clock::now();
    const QsaShapes s = shapes_of(g);
    for (int t = 0; t < T; ++t) {
        h_tok_[t] = tokens[t];
        qsa_step_fill(h_step_ + t * kStepCount, pos0 + t, s);
        for (int64_t h = 0; h < g.n_head; ++h) h_pos_[t * g.n_head + h] = (int32_t) (pos0 + t);
        for (int64_t h = 0; h < g.n_head_kv; ++h) h_pos_kv_[t * g.n_head_kv + h] = (int32_t) (pos0 + t);
        for (int64_t h = 0; h < g.idx_q_heads; ++h) h_pos_iq_[t * g.idx_q_heads + h] = (int32_t) (pos0 + t);
    }
    if (ss.ple.ready()) {
        uint32_t rows[kVerifyMaxT * PLE_N_HEADS];
        int32_t prev[2] = {ss.ple_prev[0], ss.ple_prev[1]};
        for (int t = 0; t < T; ++t) {
            ngram_rows(&tokens[t], prev, 1, ss.ple.consts, rows + t * PLE_N_HEADS);
            prev[0] = prev[1];
            prev[1] = tokens[t];
        }
        if (!ss.ple.table->gather_batch(rows, (size_t) T, h_ple_, err)) return false;
    }
    *(volatile uint32_t*) h_seq_ = 0;
    *(volatile uint32_t*) h_flag_ = 0;
    *(volatile uint32_t*) h_flagA_ = 0;
    *(volatile uint32_t*) h_flagB_ = 0;
    *(volatile uint32_t*) h_epA_ = 0;
    *(volatile uint32_t*) h_epB_ = 0;
    h_eplan_[2] = h_eplan_[plan_i32_ + 2] = 0;   // no block belongs to a layer step of this window yet
    ep_want_ = 0;
    for (uint32_t* f : h_hand_flag_) *(volatile uint32_t*) f = 0;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    last_t_ = T;
    last_pos0_ = pos0;
    for (int t = 0; t < T; ++t) last_tokens_[t] = tokens[t];
    ms_host += ms_since(t0);
    VDBG("staged; launching\n");
    // every stage's graph at once: a later stage waits on its GPU for the hand-off of the one before it
    for (Part& P : stages_) {
        DeviceGuard dg(P.device);
        const cudaError_t le = cudaGraphLaunch(P.exec[T], P.cs);
        if (le != cudaSuccess) { err = std::string("verify: launch: ") + cudaGetErrorString(le); return false; }
        (void) cudaStreamQuery(P.cs);
    }
    VDBG("launched\n");
    volatile uint32_t* const seq = h_seq_;
    volatile uint32_t* const flag = h_flag_;
    const int G = groups_[T] > 0 ? groups_[T] : 1;
    const int gtb[2] = {0, (T + 1) / 2}, gte[2] = {G == 2 ? (T + 1) / 2 : T, T};
    for (int64_t k = 0; k < g.n_layers * G; ++k) {
        const int64_t l = k / G;
        const int grp = (int) (k % G);
        const uint32_t want = (uint32_t) (k + 1);
        const Clock::time_point a = Clock::now();
        auto last_flush = a;
        uint32_t spins = 0;
        progress_at("verify window: waiting for the GPU to reach layer", l);
        while (*seq < want) {
            _mm_pause();
            if ((++spins & 1023u) != 0) continue;
            const auto now = Clock::now();
            if (now - last_flush > std::chrono::microseconds(2000)) {
                last_flush = now;
                const cudaError_t q = cudaStreamQuery(part_of(l).cs);
                if (q != cudaErrorNotReady && *seq < want) {
                    err = "verify: layer " + std::to_string(l) + " never rang (" +
                          (q == cudaSuccess ? std::string("graph finished") : std::string(cudaGetErrorString(q))) + ")";
                    return false;
                }
            }
            if (now - a > std::chrono::seconds(20)) { err = "verify: timed out at layer " + std::to_string(l); return false; }
        }
        const Clock::time_point b = Clock::now();
        VDBG("layer %lld rang\n", (long long) l);
        cur_layer_ = want - 1;
        set_plan_slot(l, grp);
        const int tb = gtb[grp], n = gte[grp] - gtb[grp];
        progress_at("verify window: the CPU experts of layer", l);
        if (pool != nullptr)
            pool(user, h_x_ + (size_t) tb * g.n_embd, h_ids_ + (size_t) tb * ss.k, n, ss.k,
                 h_ymiss_ + (size_t) tb * ss.k * g.n_embd, l);
        VDBG("layer %lld served\n", (long long) l);
        progress_tick();
        std::atomic_thread_fence(std::memory_order_seq_cst);
        _mm_sfence();
        if (*(volatile uint32_t*) h_flagA_ != want) {        // the pool did not publish a plan: an empty one
            sink_.counts[0] = 0;
            sink_.counts[1] = 0;
            sink_.counts[2] = 0;
            sink_.start[0] = 0;
            sink_.start2[0] = 0;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            *(volatile uint32_t*) h_flagA_ = want;
            raise_flag(h_flagB_, want);
            *(volatile uint32_t*) h_epA_ = want;
        }
        if (ep_want_ == want) {                              // the first GPU computes some of this layer's experts
            const Clock::time_point c = Clock::now();
            while (*(volatile uint32_t*) h_epB_ < want) {
                _mm_pause();
                if (Clock::now() - c > std::chrono::seconds(20)) {
                    err = "verify: the first GPU never served layer " + std::to_string(l);
                    return false;
                }
            }
            ms_ep += ms_since(c);
        }
        *flag = want;
        ms_wait += std::chrono::duration<double, std::milli>(b - a).count();
        ms_pool += ms_since(b);
    }
    *(volatile uint32_t*) h_epA_ = (uint32_t) (g.n_layers * G);   // the first GPU's remaining layers: nothing to serve
    progress_at("verify window: waiting for the GPU to finish the window (flags A/B/M raised)", (int64_t) T);
    for (Part& P : stages_) {
        const cudaError_t se = cudaStreamSynchronize(P.cs);
        if (se != cudaSuccess) { err = std::string("verify: ") + cudaGetErrorString(se); return false; }
    }
    progress_at("verify window: waiting for the expert copies", (int64_t) T);
    for (Part& P : stages_) cudaStreamSynchronize(P.copy);   // no host function of this window may raise flag B in the next one
    Part& L = stages_.back();
    // ---- a sampled or penalized request: the head's sampling again, host-side so its parameters are this call's
    // own (a captured kernel would replay the same draws forever).  Row t's draw is Philox(seed, pos0 + t): tied to
    // the POSITION it samples, not to how the text was cut into windows, so a seed replays the same text whatever
    // the drafts were. Exact: a rejected row's draw is discarded, and no kept decision depends on a reused draw.
    const bool sampled = !sampling_.greedy && sampling_.temperature > 0.0f;
    if (head_sampling_ && (sampled || hist_d_ != nullptr)) {
        SamplerParams sp = sampling_;
        sp.counter = (uint64_t) pos0;
        DeviceGuard dg(L.device);
        sample_tokens(L.head_logits_, T, (int) n_vocab_, hist_d_, hist_len_, sp, m_out_, L.cs);
        if (cudaStreamSynchronize(L.cs) != cudaSuccess) {   // m_out_ is the mapped h_out_: synced, it is readable
            err = "verify: the head sampling failed";
            return false;
        }
    }
    for (int t = 0; t < T; ++t) out[t] = ((volatile int32_t*) h_out_)[t];
    if (static const bool dbg = std::getenv("STRATA_DBG_NAN") != nullptr; dbg) {   // debug: the first non-finite head
        static bool reported = false;
        if (!reported) {
            std::vector<float> h((size_t) T * (size_t) n_vocab_);
            cudaMemcpy(h.data(), L.head_logits_, h.size() * 4, cudaMemcpyDeviceToHost);
            for (int t = 0; t < T && !reported; ++t) {
                int64_t bad = 0;
                for (int64_t v = 0; v < n_vocab_; ++v) bad += !std::isfinite(h[(size_t) t * n_vocab_ + v]);
                if (bad) {
                    reported = true;
                    std::fprintf(stderr, "strata dbg: verify window at position %lld, row %d: %lld of %lld logits non-finite "
                                         "(token out %d)\n", (long long) pos0, t, (long long) bad, (long long) n_vocab_, out[t]);
                }
            }
        }
    }
    VDBG("window done\n");
    ++windows;
    progress_at("decode");
    progress_beat();
    return true;
}

void Verifier::set_plan_slot(int64_t layer, int grp) {
    cur_part_ = &part_of(layer);
    const int64_t cap = sink_.cap;
    int32_t* base = h_plan_ + (size_t) grp * (size_t) plan_i32_;
    const int64_t i32 = 4 + (cap + 1) + cap + cap;
    const int64_t ptr_off = (i32 + 1) & ~1ll;
    sink_.counts = base;
    sink_.start = base + 4;
    sink_.dst = sink_.start + cap + 1;
    sink_.tok = sink_.dst + cap;
    sink_.ptr = (unsigned long long*) (base + ptr_off);
    sink_.ptr2 = sink_.ptr + cap;
    sink_.start2 = base + ptr_off + 4 * cap;
    const int G = groups_[last_t_] > 0 ? groups_[last_t_] : 1;
    const int64_t per = G == 2 ? kStagingBlobs / 2 : kStagingBlobs;
    sink_.staging = (unsigned long long) (cur_part_->staging_ +
                                          (size_t) (grp * per) * strata::kernels::cpu::expert_layout().max_blob);
    sink_.staging_cap = per;
    if (ep_) {
        int32_t* eb = h_eplan_ + (size_t) grp * (size_t) plan_i32_;
        sink_.ep_counts = eb;
        sink_.ep_start = eb + 4;
        sink_.ep_dst = sink_.ep_start + cap + 1;
        sink_.ep_tok = sink_.ep_dst + cap;
        sink_.ep_ptr = (unsigned long long*) (eb + ptr_off);
    }
}

void Verifier::set_ep(int32_t slot_end) {
    ep_ = true;
    sink_.ep_slot_end = slot_end;
    sink_.ep_layer0 = stages_.size() > 1 ? stages_[1].l0 : g_->n_layers;
}

// Flag B only rises: a host function of an earlier layer may run after a later layer already raised it directly.
void Verifier::raise_flag(uint32_t* flag, uint32_t value) {
    volatile long* f = (volatile long*) flag;
#if defined(_WIN32)
    long cur = *f;
    while ((uint32_t) cur < value) {
        const long prev = _InterlockedCompareExchange(f, (long) value, cur);
        if (prev == cur) break;
        cur = prev;
    }
#else
    uint32_t cur = __atomic_load_n((uint32_t*) flag, __ATOMIC_SEQ_CST);
    while (cur < value && !__atomic_compare_exchange_n((uint32_t*) flag, &cur, value, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {}
#endif
}

// Plan v0.3 P6: the PCIe share by DMA.  The copy engine moves the blobs while the CPU computes its own share and the
// GPU its VRAM experts; a host function raises flag B when they have landed (the graph waits for it before the PCIe
// groups).  Staging is split between the two token groups of a split window.
void Verifier::fetch_dma(void* ctx, const uint8_t* const* src, int n, size_t bytes) {
    Verifier* v = (Verifier*) ctx;
    const uint32_t want = v->cur_layer_ + 1;
    if (n <= 0) { raise_flag(v->h_flagB_, want); return; }
    uint8_t* stage = (uint8_t*) v->sink_.staging;                  // this group's half in a split window
    const Part& P = *v->cur_part_;                                   // the layer's stage: its GPU and copy engine
    DeviceGuard dg(P.device);
    for (int i = 0; i < n; ++i) cudaMemcpyAsync(stage + (size_t) i * bytes, src[i], bytes, cudaMemcpyHostToDevice, P.copy);
    FlagSet& fs = v->flag_sets_[v->cur_layer_ % (sizeof v->flag_sets_ / sizeof v->flag_sets_[0])];
    fs.flag = v->h_flagB_;
    fs.value = want;
    cudaLaunchHostFunc(P.copy, [](void* p) { FlagSet* s = (FlagSet*) p; raise_flag(s->flag, s->value); }, &fs);
}

void Verifier::publish_plan(void* ctx) {
    Verifier* v = (Verifier*) ctx;
    const uint32_t want = v->cur_layer_ + 1;
    if (v->sink_.ep_counts != nullptr) {   // every layer step, so the first GPU keeps pace with the pool
        if (v->sink_.ep_counts[0] > 0) {
            v->sink_.ep_counts[2] = (int32_t) want;
            v->ep_want_ = want;
        }
        _mm_sfence();
        *(volatile uint32_t*) v->h_epA_ = want;
    }
    _mm_sfence();
    *(volatile uint32_t*) v->h_flagA_ = want;
}

bool Verifier::commit(int n_keep, std::string& err) { return commit_launch(n_keep, err) && commit_wait(err); }

bool Verifier::commit_launch(int n_keep, std::string& err) {
    if (n_keep < 1 || n_keep > last_t_) { err = "verify: commit count out of range"; return false; }
    const Clock::time_point t0 = Clock::now();
    h_commit_[0] = n_keep;
    h_commit_[1] = n_keep - 1;
    for (int t = 0; t < max_t_; ++t) h_commit_[2 + t] = t < n_keep ? (int32_t) (last_pos0_ + t) : -1;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    for (Part& P : stages_) {
        DeviceGuard dg(P.device);
        const cudaError_t le = cudaGraphLaunch(P.commit_exec, P.cs);
        if (le != cudaSuccess) { err = std::string("verify: commit launch: ") + cudaGetErrorString(le); return false; }
    }
    for (int t = 0; t < n_keep; ++t) {
        ss_->ple_prev[0] = ss_->ple_prev[1];
        ss_->ple_prev[1] = last_tokens_[t];
    }
    ms_commit += ms_since(t0);
    return true;
}

bool Verifier::commit_wait(std::string& err) {
    const Clock::time_point t0 = Clock::now();
    for (Part& P : stages_) {
        const cudaError_t se = cudaStreamSynchronize(P.cs);
        if (se != cudaSuccess) { err = std::string("verify: commit: ") + cudaGetErrorString(se); return false; }
    }
    ms_commit += ms_since(t0);
    return true;
}

}  // namespace strata::core
