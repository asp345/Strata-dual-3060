// src/prefill/prefill.cpp - see include/strata/prefill/prefill.hpp.
#include "strata/prefill/prefill.hpp"
#include "strata/core/progress.hpp"

#include "strata/core/layout.hpp"
#include "strata/core/placement.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/ple.hpp"
#include "strata/kernels/native_ple_postops.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/cvec.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/core/layer.hpp"
#include "strata/core/native_head.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_prompt_attn.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include "strata/prefill/kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#ifndef STRATA_PREFILL_MMQ
// A build without the llama.cpp sources (no STRATA_NATIVE_EXPERTS): no MMQ, the FP16 expert path everywhere.
namespace strata::prefill::mmq {
bool built() { return false; }
bool supported(int) { return false; }
size_t matrix_bytes(int, int64_t, int64_t) { return 0; }
size_t q8_bytes(int64_t, int64_t) { return 0; }
void quantize(const float*, const int32_t*, void*, int, int64_t, int64_t, int64_t, void*) {}
Context::Context() {}
Context::~Context() {}
void Context::run(const Product&, void*) {}
void gather_native(const void*, const void*, size_t, const void*, size_t, void*, void*, void*) {}
void gather_strata_q2(const uint8_t*, void*, void*, void*) {}
void swiglu(const float*, float*, int64_t, int64_t, bool, void*) {}
void iota(int32_t*, int64_t, void*) {}
}  // namespace strata::prefill::mmq
#endif

namespace strata::prefill {
namespace {

using Clock = std::chrono::steady_clock;
constexpr float EPS = 1e-6f;
constexpr int64_t N = 2560, HC = 4, D = N * HC, LR = 320, K = 10, NE = 512;
constexpr int64_t C = 10240, ZV = 6144, HV = 48;
// plan v0.3 P6: staging holds the largest blob of the pack (a native pack's blobs differ per layer)
inline int64_t MAXBLOB() { return (int64_t) strata::kernels::cpu::expert_layout().max_blob; }
constexpr int STAGE = 8;           // host->device expert staging ring (chunks below STREAM_ALL_MIN)
// Step 3: from this chunk size on, every non-resident expert of every layer streams in a fixed order through a
// RING_MAX-slot ring (nearly all 512 are routed at such a chunk), so the copy engine keeps working through the
// attention halves instead of waiting for each layer's routing.
constexpr int RING_MAX = 512;           // the arrays; the ring itself is ring_slots()
constexpr int64_t STREAM_ALL_MIN = 2048;
double g_pinned_share = 1.0;
// The streamed ring: 384 slots at 8192-token chunks when (nearly) every streamed expert is DMA'd from pinned RAM -
// measured on Q2_0: 96 slots 1153 tok/s, 384 1294 (the next layer's experts arrive during its attention half) -
// and 96 when a large share goes through host copies (IQ3_S on 64 GB, a third unpinned: 96 slots 1216, 256 1070 -
// the host copies are the limit and the bigger ring only takes cache slots).  The attention half, and so the ring
// it needs, scales with the chunk: 384 * T / 8192, at least 96 (IQ3_XXS on two GPUs, one on a x4 link: 6.8K prompt
// in 3584-token chunks, 384 slots 7.58 / 7.62 s, 128 7.37 / 7.36 s).  STRATA_PREFILL_RING overrides.
inline int ring_slots(size_t T) {
    const char* v = std::getenv("STRATA_PREFILL_RING");
    const int r = v ? std::atoi(v)
                    : g_pinned_share >= 0.9 ? (int) std::max<int64_t>(96, 384 * (int64_t) T / 8192) : 96;
    const int big = r < 16 ? 16 : r > RING_MAX ? RING_MAX : r;
    return (int64_t) T >= STREAM_ALL_MIN ? big : STAGE;
}
constexpr int DQ = 2;              // dequantized-expert ring (FP16 gate/up + down)

double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

// Either cudaMalloc (owned, freed with the object) or a bump allocation from a borrowed region; with no base and
// no region it only counts, which is how `bytes_needed` sizes the region.
struct Alloc {
    uint8_t* base = nullptr;
    uint64_t cap = 0, used = 0;
    bool count_only = false;
    std::vector<void*>* owned = nullptr;
    template <typename T> T* take(size_t n, bool& ok) {
        const uint64_t bytes = ((uint64_t) n * sizeof(T) + 256 + 255) & ~255ull;
        if (count_only) { used += bytes; return nullptr; }
        if (base != nullptr) {
            if (used + bytes > cap) { ok = false; return nullptr; }
            T* p = (T*) (base + used);
            used += bytes;
            return p;
        }
        void* p = nullptr;
        if (cudaMalloc(&p, bytes) != cudaSuccess) { ok = false; return nullptr; }
        owned->push_back(p);
        used += bytes;
        return (T*) p;
    }
};

}  // namespace

// Step 4 of the prompt-speed plan: the experts the arena could not pin (a third of the streamed ones on IQ3_S) are
// copied into pinned buffers by these threads, ahead of the launches.  Copied in line by the launching thread they
// left the GPU without queued work while each ~2 MB memcpy ran (~15 s of a 32K prompt on IQ3_S).  Job j - a layer's
// j-th unpinned expert, in launch order - lands in host buffer j % kRing, which is free again once the DMA of job
// j - kRing (recorded by the launching thread, `issued`) is done.
struct Stager {
    static constexpr int kRing = 16;
    struct Job { const uint8_t* src; size_t bytes; };
    uint8_t* buf[kRing] = {};
    bool pinned[kRing] = {};
    std::vector<std::vector<uint8_t>> pageable;   // the fallback when no more RAM can be pinned
    cudaEvent_t dma_done[kRing] = {};
    std::vector<Job> jobs;
    std::unique_ptr<std::atomic<int>[]> ready;
    size_t ready_cap = 0;
    // gen << 32 | n << 16 | next index: a claim is a CAS on the generation it woke for (a thread late from the
    // previous layer can never take a job of this one - the expert pool's issue #29 lesson)
    std::atomic<uint64_t> head{0};
    std::atomic<int> issued{0}, active{0};
    uint32_t gen = 0;
    bool quit = false;
    std::mutex mu;
    std::condition_variable cv;
    std::vector<std::thread> threads;
    int device = 0;

    bool init(size_t blob_bytes, int nthreads) {
        pageable.resize(kRing);
        for (int i = 0; i < kRing; ++i) {
            pinned[i] = cudaHostAlloc((void**) &buf[i], blob_bytes, cudaHostAllocDefault) == cudaSuccess;
            if (!pinned[i]) {
                cudaGetLastError();
                pageable[(size_t) i].resize(blob_bytes);
                buf[i] = pageable[(size_t) i].data();
            }
            if (cudaEventCreateWithFlags(&dma_done[i], cudaEventDisableTiming) != cudaSuccess) return false;
        }
        cudaGetDevice(&device);
        for (int t = 0; t < nthreads; ++t) threads.emplace_back([this] { work(); });
        return true;
    }
    ~Stager() {
        finish();
        { std::lock_guard<std::mutex> lk(mu); quit = true; }
        cv.notify_all();
        for (auto& t : threads) t.join();
        for (int i = 0; i < kRing; ++i) {
            if (dma_done[i]) cudaEventDestroy(dma_done[i]);
            if (buf[i] && pinned[i]) cudaFreeHost(buf[i]);
        }
    }
    void work() {
        cudaSetDevice(device);
        uint32_t seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return quit || gen != seen; });
                if (quit) return;
                seen = gen;
            }
            for (;;) {
                active.fetch_add(1, std::memory_order_acq_rel);
                const int j = claim(seen);
                if (j < 0) { active.fetch_sub(1, std::memory_order_acq_rel); break; }
                const int b = j % kRing;
                if (j >= kRing) {
                    while (issued.load(std::memory_order_acquire) <= j - kRing) std::this_thread::yield();
                    cudaEventSynchronize(dma_done[b]);
                }
                std::memcpy(buf[b], jobs[(size_t) j].src, jobs[(size_t) j].bytes);
                ready[(size_t) j].store(1, std::memory_order_release);
                active.fetch_sub(1, std::memory_order_acq_rel);
            }
        }
    }
    int claim(uint32_t g) {
        uint64_t cur = head.load(std::memory_order_acquire);
        for (;;) {
            if ((uint32_t) (cur >> 32) != g) return -1;
            const int n = (int) ((cur >> 16) & 0xffff), j = (int) (cur & 0xffff);
            if (j >= n) return -1;
            if (head.compare_exchange_weak(cur, cur + 1, std::memory_order_acq_rel, std::memory_order_acquire)) return j;
        }
    }
    /// A layer's jobs; the previous layer's are finished (finish()).
    void start(std::vector<Job>&& js) {
        if (js.empty()) return;
        std::lock_guard<std::mutex> lk(mu);
        jobs = std::move(js);
        if (ready_cap < jobs.size()) {
            ready_cap = jobs.size() * 2;
            ready.reset(new std::atomic<int>[ready_cap]);
        }
        for (size_t i = 0; i < jobs.size(); ++i) ready[i].store(0, std::memory_order_relaxed);
        issued.store(0);
        ++gen;
        head.store((uint64_t) gen << 32 | (uint64_t) jobs.size() << 16, std::memory_order_release);
        cv.notify_all();
    }
    /// Job j's bytes, in a pinned buffer (waits for the copy).
    const uint8_t* wait(int j) {
        while (!ready[(size_t) j].load(std::memory_order_acquire)) std::this_thread::yield();
        return buf[j % kRing];
    }
    /// The launching thread queued job j's DMA on `copy`: its buffer is free once that is done.
    void issued_one(int j, cudaStream_t copy) {
        cudaEventRecord(dma_done[j % kRing], copy);
        issued.store(j + 1, std::memory_order_release);
    }
    /// No job is running after this (the end of a layer, or an early return in the middle of one).
    void finish() {
        head.store((uint64_t) gen << 32, std::memory_order_release);   // n = 0: nothing more to claim
        issued.store(1 << 30, std::memory_order_release);
        while (active.load(std::memory_order_acquire) != 0) std::this_thread::yield();
    }
};

struct Prefill::Impl {
    const core::WeightTable* wt = nullptr;
    const core::ModelGeometry* g = nullptr;
    core::SessionState* ss = nullptr;
    core::ExpertSource* src = nullptr;
    const uint64_t* slot_addr = nullptr;
    const int32_t* host_res = nullptr;
    int32_t slot_lo = 0, slot_hi = 0;   // this GPU's slots
    bool resident(int64_t l, int32_t e) const {
        if (host_res == nullptr || slot_addr == nullptr) return false;
        const int32_t s = host_res[(size_t) l * g->n_expert + e];
        return s >= slot_lo && s < slot_hi;
    }
    // this part's pipeline stage (placement.hpp): its GPU, its layers [l0, l1), their first GDN / QSA index
    int stage = 0, device = 0;
    int64_t l0 = 0, l1 = 0, gdn0 = 0, qsa0 = 0;
    bool has_qsa = false;
    cudaEvent_t done = nullptr, handed = nullptr;   // this stage's layers of a chunk ran; its residual was copied on
    cudaEvent_t xa = nullptr, xb = nullptr;         // the helper GPU's hand-offs (see run: `helper`)
    int64_t T = 0, T_max = 0;
    bool borrowed = false;
    cudaStream_t cs = nullptr, copy = nullptr;
    Gemm gemm;
    std::vector<void*> owned;
    // chunk buffers
    float *emb = nullptr, *R = nullptr, *row_rs = nullptr, *lo = nullptr, *gated = nullptr, *inj = nullptr;
    uint16_t *xn16 = nullptr, *lo16 = nullptr;
    float* mixed = nullptr;
    uint16_t *mixed_bf = nullptr, *mixed_h = nullptr;
    float* bo = nullptr;
    // GDN
    float *qkv = nullptr, *z = nullptr, *ab = nullptr, *gate = nullptr, *beta = nullptr, *hbuf = nullptr, *y = nullptr;
    uint16_t* y_h = nullptr;
    // QSA
    float *Kc = nullptr, *Vc = nullptr, *Qf = nullptr, *q = nullptr, *idx_raw = nullptr, *q_idx = nullptr, *attn = nullptr;
    uint16_t* attn_h = nullptr;
    int32_t* steps_dev = nullptr;
    std::vector<int32_t> steps_host;
    int32_t* sel_ids = nullptr;
    float* sel_scores = nullptr;          // [sel_batch, max_blocks]
    int64_t sel_batch = 256, max_blocks = 0;
    float* attn_scratch = nullptr;
    int64_t attn_batch = 32, cap = 0;
    // MoE
    float *logits = nullptr, *w = nullptr, *GU = nullptr, *Dm = nullptr, *sgate = nullptr, *sup = nullptr,
          *shared = nullptr, *sg = nullptr;
    int32_t *ids = nullptr, *slot_dev = nullptr, *src_dev = nullptr;
    uint16_t *Xs = nullptr, *Hh = nullptr, *sh_h = nullptr;
    // step 2b (MMQ): the activations quantized per layer, H in FP32 and its group's quantized rows, the identity
    // row map, the group bounds, the group buffers of gathered experts
    void *Xq = nullptr, *Hq = nullptr;
    float* H = nullptr;
    int32_t *ids_identity = nullptr, *bounds_dev = nullptr;
    uint8_t *grp_gu = nullptr, *grp_d = nullptr;
    std::vector<int32_t> bounds_host;
    std::unique_ptr<mmq::Context> mmq_ctx;
    std::vector<int32_t> ids_host, slot_host, src_host, cnt, off;
    uint16_t* dq_gu[DQ] = {};
    uint16_t* dq_d[DQ] = {};
    uint8_t* stage_dev[RING_MAX] = {};
    int ring = STAGE;                        // the slots of this layout's ring (ring_slots)
    std::unique_ptr<Stager> stager;          // the unpinned experts' host copies (step 4)
    cudaEvent_t copied[RING_MAX] = {}, used[RING_MAX] = {};
    bool stage_live[RING_MAX] = {};
    // PLE
    float* ple_emb = nullptr;
    std::vector<float> ple_pageable[2];      // the fallback when no more RAM can be pinned
    float* ple_emb_host[2] = {};             // pinned, double-buffered: the next chunk's rows are read while this
    cudaEvent_t ple_copied[2] = {};          // one runs; the event marks that buffer's upload done
    std::vector<uint32_t> ple_rows[2];
    float* ple_norm = nullptr;
    int32_t* tok_dev = nullptr;              // the first stage: a chunk's token ids, for one embedding gather
    std::vector<int32_t> tok_host;
    uint8_t* region = nullptr;               // the attention/MoE scratch region (idle while the PLE block runs)
    uint64_t region_bytes = 0;
    PrefillStats* stats = nullptr;
    // KV streaming: one layer's whole K/V, staged from the host copy per layer and chunk (identity layout)
    strata::kernels::KvHostPools stage_pool;
    int32_t* ident_table = nullptr;
};

namespace {
// the staging pool of a streamed session: every page of one layer (same sequence in init and bytes_needed)
// STRATA_KV_STAGE_OWN (A/B only): the staging pool gets its own allocation instead of borrowed expert slots, so a
// streamed run lends the prompt path exactly the slots a resident one does (a lent expert runs on the CPU, which
// rounds differently: without this an A/B compares two expert placements as well as two KV placements)
bool stage_own() { static const bool v = std::getenv("STRATA_KV_STAGE_OWN") != nullptr; return v; }
void take_stage(Alloc& o_borrowed, const core::SessionState& ss, const strata::kernels::QsaShapes& s, bool has_qsa,
                strata::kernels::KvHostPools& st, bool& ok) {
    const core::QsaState& q0 = ss.qsa_states[0];
    if (q0.kv_mode != 1 || !has_qsa) return;
    if (stage_own() && o_borrowed.count_only) return;
    Alloc own;
    own.owned = o_borrowed.owned;
    Alloc& o = stage_own() ? own : o_borrowed;
    const size_t rows = (size_t) q0.n_pages * s.n_head_kv * s.page_size;
    if (q0.kv_q4) {
        st.k_q4 = o.take<uint8_t>(rows * strata::kernels::kv_q4_bytes_per_head((int) s.head_dim), ok);
        st.v_q4 = o.take<uint8_t>(rows * strata::kernels::kv_q4_bytes_per_head((int) s.head_dim), ok);
    } else if (q0.kv_int8) {
        st.k_q = o.take<int8_t>(rows * s.head_dim, ok);
        st.v_q = o.take<int8_t>(rows * s.head_dim, ok);
        st.k_scale = o.take<uint16_t>(rows * (s.head_dim / 64), ok);
        st.v_scale = o.take<uint16_t>(rows * (s.head_dim / 64), ok);
    } else {
        st.k_pool = o.take<uint16_t>(rows * s.head_dim, ok);
        st.v_pool = o.take<uint16_t>(rows * s.head_dim, ok);
    }
}
strata::kernels::QsaAttnPools pools_of(const strata::kernels::KvHostPools& h, const int32_t* table) {
    strata::kernels::QsaAttnPools p;
    p.k_pool = h.k_pool; p.v_pool = h.v_pool; p.k_q = h.k_q; p.v_q = h.v_q; p.k_scale = h.k_scale; p.v_scale = h.v_scale;
    p.k_q4 = h.k_q4; p.v_q4 = h.v_q4;
    p.page_table = table;
    return p;
}
}  // namespace

Prefill::Prefill() = default;
Prefill::~Prefill() {
    for (auto& part : parts_) {
        Impl& m = *part;
        core::DeviceGuard dg(m.device);
        if (m.cs) cudaStreamSynchronize(m.cs);
        for (int i = 0; i < RING_MAX; ++i) {
            if (m.copied[i]) cudaEventDestroy(m.copied[i]);
            if (m.used[i]) cudaEventDestroy(m.used[i]);
        }
        for (int b = 0; b < 2; ++b) {
            if (m.ple_copied[b]) cudaEventDestroy(m.ple_copied[b]);
            if (m.ple_emb_host[b] && m.ple_pageable[b].empty()) cudaFreeHost(m.ple_emb_host[b]);
        }
        if (m.done) cudaEventDestroy(m.done);
        if (m.xa) cudaEventDestroy(m.xa);
        if (m.xb) cudaEventDestroy(m.xb);
        if (m.handed) cudaEventDestroy(m.handed);
        m.stager.reset();
        m.mmq_ctx.reset();
        if (m.copy) cudaStreamDestroy(m.copy);
        if (m.cs) cudaStreamDestroy(m.cs);
        for (void* p : m.owned) cudaFree(p);
    }
}

namespace {
constexpr int64_t GEMM_SCRATCH = 32ll << 20;        // FP16 elements for the largest dequantized dense weight
constexpr size_t GEMM_WS = 32u << 20;               // cuBLAS workspace

// THE ATTENTION HALF AND THE MoE HALF SHARE THEIR BUFFERS.  A layer runs its attention (GDN or QSA), writes it back
// into the residual, and only then its MoE, so the three sets of scratch are never live at once: one region the size
// of the largest holds them all.  That is ~260 KB of the ~680 KB a prompt token cost - which is what lets a chunk
// grow (every expert is streamed once per chunk, so a bigger chunk streams fewer bytes per token).  The sizes are
// counted with the same `take` sequence `init` uses; a mismatch makes `init` fail with "do not fit", never overlap.
uint64_t gdn_set_bytes(size_t T) {
    Alloc a; a.count_only = true; bool ok = true;
    a.take<float>(T * C, ok); a.take<float>(T * ZV, ok); a.take<float>(T * 2 * HV, ok); a.take<float>(T * HV, ok);
    a.take<float>(T * HV, ok); a.take<float>(T * C, ok); a.take<float>(T * ZV, ok); a.take<uint16_t>(T * ZV, ok);
    return a.used;
}
uint64_t qsa_set_bytes(size_t T, int64_t cap, int64_t max_blocks, int64_t sel_batch, int64_t attn_batch,
                       const strata::kernels::QsaShapes& s) {
    Alloc a; a.count_only = true; bool ok = true;
    a.take<float>(T * 512, ok); a.take<float>(T * 512, ok); a.take<float>(T * 12288, ok); a.take<float>(T * ZV, ok);
    a.take<float>(T * 128, ok); a.take<float>(T * 512, ok); a.take<float>(T * ZV, ok); a.take<uint16_t>(T * ZV, ok);
    a.take<int32_t>(T * (size_t) cap, ok);
    a.take<float>((size_t) sel_batch * (size_t) max_blocks, ok);
    a.take<float>((size_t) attn_batch * strata::kernels::qsa_decode_attn_scratch_floats(cap, s), ok);
    return a.used;
}
// Step 2b: which layers' experts go through MMQ (both weight types covered; the Strata Q2_0 pack always - its blob
// is converted to GGUF Q2_0 blocks on the gather), whether any layer keeps the FP16 path (IQ1_M), and the largest
// gate/up and down matrices a group buffer slot holds.  STRATA_PREFILL_MMQ=0: the FP16 path everywhere (the A/B).
constexpr int MMQ_GROUP = 16;                  // experts per MMQ launch (the gather is per expert, as blobs arrive)
// MMQ reads up to one 256-value tile past a matrix's last row when the row length is not a multiple of it (the down
// product: 640 values).  Those bytes meet zero activations, which is harmless only if they decode to finite numbers -
// llama.cpp zero-pads after every tensor, and so does a group buffer: this many zeroed bytes follow its last expert.
constexpr size_t MMQ_TAIL = 4096;
struct MmqPlan {
    bool any = false, fallback = true;
    std::vector<char> layer;                   // per layer: MMQ
    size_t gu_max = 0, d_max = 0;
};
const MmqPlan& mmq_plan() {
    static const MmqPlan plan = [] {
        MmqPlan p;
        const auto& lay = strata::kernels::cpu::expert_layout();
        const char* env = std::getenv("STRATA_PREFILL_MMQ");
        const bool on = mmq::built() && (env == nullptr || std::atoi(env) != 0);
        const int64_t layers = lay.native ? (int64_t) lay.fmt.size() : lay.n_layers;
        p.layer.assign((size_t) std::max<int64_t>(layers, 0), 0);
        p.fallback = !on || layers <= 0;
        for (int64_t l = 0; on && l < layers; ++l) {
            const int gt = lay.native ? lay.fmt[(size_t) l].gu_type : 42, dt = lay.native ? lay.fmt[(size_t) l].d_type : 42;
            if (!mmq::supported(gt) || !mmq::supported(dt)) { p.fallback = true; continue; }
            p.layer[(size_t) l] = 1;
            p.any = true;
            p.gu_max = std::max(p.gu_max, mmq::matrix_bytes(gt, 1280, N));
            p.d_max = std::max(p.d_max, mmq::matrix_bytes(dt, N, 640));
        }
        return p;
    }();
    return plan;
}
uint64_t moe_set_bytes(size_t T, int64_t n_expert) {
    const MmqPlan& mp = mmq_plan();
    Alloc a; a.count_only = true; bool ok = true;
    a.take<float>(T * n_expert, ok); a.take<float>(T * K, ok); a.take<int32_t>(T * K, ok); a.take<int32_t>(T * K, ok);
    a.take<int32_t>(T * K, ok);
    if (mp.fallback) a.take<uint16_t>(T * K * N, ok);
    a.take<float>(T * K * 1280, ok);
    if (mp.fallback) a.take<uint16_t>(T * K * 640, ok);
    a.take<float>(T * K * N, ok); a.take<float>(T * 640, ok);
    a.take<float>(T * 640, ok); a.take<uint16_t>(T * 640, ok); a.take<float>(T * N, ok); a.take<float>(T, ok);
    if (mp.any) {
        a.take<uint8_t>(mmq::q8_bytes((int64_t) (T * K), N), ok);
        a.take<float>(T * K * 640, ok);
        a.take<uint8_t>(mmq::q8_bytes((int64_t) (T * K), 640), ok);
    }
    return a.used;
}
}

bool Prefill::init(const core::WeightTable& wt, const core::ModelGeometry& g, core::SessionState& ss,
                   core::ExpertSource* src, const uint64_t* slot_addr, const int32_t* host_res, const int32_t* slot_first,
                   int64_t chunk, std::string& err, const std::vector<Borrow>& borrow) {
    if (g.n_embd != N || g.hc != HC || g.hc_lr != LR || g.n_expert < 1 || ss.k != K) {
        err = "prefill: geometry differs from the artifact's"; return false;
    }
    const core::Placement& pl = core::placement();
    if (!borrow.empty() && borrow.size() != (size_t) pl.stages()) { err = "prefill: one borrowed region per GPU"; return false; }
    const size_t T = (size_t) chunk;
    int64_t gdn = 0, qsa = 0;
    for (int st = 0; st < pl.stages(); ++st) {
        parts_.push_back(std::make_unique<Impl>());
        Impl& m = *parts_.back();
        m.wt = &wt; m.g = &g; m.ss = &ss; m.src = src; m.slot_addr = slot_addr; m.host_res = host_res;
        if (slot_first != nullptr) { m.slot_lo = slot_first[st]; m.slot_hi = slot_first[st + 1]; }
        m.T = chunk; m.stats = &stats_;
        m.stage = st;
        m.device = pl.device(st);
        m.l0 = pl.first(st);
        m.l1 = pl.end(st);
        m.gdn0 = gdn;
        m.qsa0 = qsa;
        for (int64_t l = m.l0; l < m.l1; ++l) {
            if (core::is_qsa_layer(g, l)) ++qsa;
            else ++gdn;
        }
        m.has_qsa = qsa > m.qsa0;
        const bool ple = pl.stage_of(1) == st;
        core::DeviceGuard dg(m.device);
        if (cudaStreamCreateWithFlags(&m.copy, cudaStreamNonBlocking) != cudaSuccess ||
            cudaStreamCreateWithFlags(&m.cs, cudaStreamNonBlocking) != cudaSuccess) {
            err = "prefill: streams"; return false;
        }
        m.T_max = chunk;
        m.borrowed = !borrow.empty() && borrow[(size_t) st].base != nullptr;
        bool ok = true;
        // one-time: events, the stager, the host buffers (for the largest chunk), the identity page table
        for (int i = 0; i < RING_MAX; ++i) {
            if (cudaEventCreateWithFlags(&m.copied[i], cudaEventDisableTiming) != cudaSuccess) ok = false;
            if (cudaEventCreateWithFlags(&m.used[i], cudaEventDisableTiming) != cudaSuccess) ok = false;
        }
        if (cudaEventCreateWithFlags(&m.done, cudaEventDisableTiming) != cudaSuccess ||
            cudaEventCreateWithFlags(&m.handed, cudaEventDisableTiming) != cudaSuccess ||
            cudaEventCreateWithFlags(&m.xa, cudaEventDisableTiming) != cudaSuccess ||
            cudaEventCreateWithFlags(&m.xb, cudaEventDisableTiming) != cudaSuccess)
            ok = false;
        m.stager = std::make_unique<Stager>();
        const int hw = (int) std::thread::hardware_concurrency();
        if (!m.stager->init((size_t) MAXBLOB(), std::max(2, std::min(4, hw / 4)))) ok = false;
        m.steps_host.resize(T * strata::kernels::kStepCount);
        m.ids_host.resize(T * K); m.slot_host.resize(T * K); m.src_host.resize(T * K); m.cnt.resize(m.g->n_expert); m.off.resize(m.g->n_expert + 1);
        for (int b = 0; b < 2 && ple; ++b) {   // the PLE rows of a chunk, read on the host for layer 1's stage
            if (cudaHostAlloc((void**) &m.ple_emb_host[b], (size_t) T * N * 4, cudaHostAllocDefault) != cudaSuccess) {
                cudaGetLastError();
                m.ple_pageable[b].resize(T * N);          // pageable: the upload is staged before it returns
                m.ple_emb_host[b] = m.ple_pageable[b].data();
            }
            if (cudaEventCreateWithFlags(&m.ple_copied[b], cudaEventDisableTiming) != cudaSuccess) ok = false;
            m.ple_rows[b].resize(T * strata::kernels::PLE_N_HEADS);
        }
        if (ss.qsa_states[0].kv_mode == 1 && m.has_qsa) {   // KV streaming: the staging pool's identity page table
            const int64_t pages = ss.qsa_states[0].n_pages;
            std::vector<int32_t> ident((size_t) pages);
            for (int64_t i = 0; i < pages; ++i) ident[(size_t) i] = (int32_t) i;
            if (cudaMalloc((void**) &m.ident_table, ident.size() * 4) != cudaSuccess ||
                cudaMemcpy(m.ident_table, ident.data(), ident.size() * 4, cudaMemcpyHostToDevice) != cudaSuccess)
                ok = false;
            else
                m.owned.push_back(m.ident_table);
        }
        if (st == 0) {
            if (cudaMalloc((void**) &m.tok_dev, T * sizeof(int32_t)) != cudaSuccess) ok = false;
            else m.owned.push_back(m.tok_dev);
            m.tok_host.resize(T);
        }
        if (!ok) { err = "prefill: host buffers or events for a chunk of " + std::to_string(chunk) + " tokens"; return false; }
        Alloc o;
        o.base = m.borrowed ? (uint8_t*) borrow[(size_t) st].base : nullptr;
        o.cap = m.borrowed ? borrow[(size_t) st].bytes : 0;
        o.owned = &m.owned;
        {
            uint16_t* gs = o.take<uint16_t>((size_t) GEMM_SCRATCH, ok);
            void* ws = o.take<uint8_t>(GEMM_WS, ok);
            if (!ok) { err = "prefill: GEMM scratch does not fit"; return false; }
            if (!m.gemm.init_external(m.cs, gs, GEMM_SCRATCH, ws, GEMM_WS, err)) return false;
        }
        if (!carve(m, T, &o)) {
            err = "prefill: device buffers for a chunk of " + std::to_string(chunk) + " tokens do not fit on GPU " +
                  std::to_string(m.device);
            return false;
        }
    }
    return true;
}

// Every device buffer of a chunk of T tokens, from the Alloc `alloc` (after the GEMM scratch and workspace): `init`
// once, and `relayout` for a request's own chunk.  The order is `bytes_needed`'s.
bool Prefill::carve(Impl& m, size_t T, void* alloc) {
    Alloc& o = *static_cast<Alloc*>(alloc);
    const core::ModelGeometry& g = *m.g;
    core::SessionState& ss = *m.ss;
    bool ok = true;
    m.emb = o.take<float>(T * N, ok); m.R = o.take<float>(T * D, ok); m.row_rs = o.take<float>(T * HC, ok);
    m.xn16 = o.take<uint16_t>(T * D, ok); m.lo = o.take<float>(T * LR, ok); m.lo16 = o.take<uint16_t>(T * LR, ok);
    m.gated = o.take<float>(T * D, ok); m.inj = o.take<float>(T * HC, ok);
    m.mixed = o.take<float>(T * N, ok); m.mixed_bf = o.take<uint16_t>(T * N, ok);
    m.mixed_h = o.take<uint16_t>(T * N, ok); m.bo = o.take<float>(T * N, ok);
    m.steps_dev = o.take<int32_t>(T * strata::kernels::kStepCount, ok);
    strata::kernels::QsaShapes s = strata::kernels::qsa_real_shapes();
    s.n_head = g.n_head; s.n_head_kv = g.n_head_kv; s.head_dim = g.head_dim; s.idx_n_head = g.idx_q_heads;
    s.idx_dim = g.idx_key_dim;
    m.cap = strata::kernels::qsa_selection_width(strata::kernels::kTopkMaxCells, s);
    m.max_blocks = ss.qsa_states[0].max_cells / s.idx_block + 2;
    {
        // one region for the attention half's and the MoE half's scratch (see gdn_set_bytes)
        const uint64_t region = std::max({gdn_set_bytes(T), qsa_set_bytes(T, m.cap, m.max_blocks, m.sel_batch,
                                                                           m.attn_batch, s), moe_set_bytes(T, m.g->n_expert)});
        uint8_t* base = o.take<uint8_t>((size_t) region, ok);
        m.region = base;
        m.region_bytes = region;
        Alloc a;
        a.base = base; a.cap = region; a.owned = &m.owned;
        m.qkv = a.take<float>(T * C, ok); m.z = a.take<float>(T * ZV, ok); m.ab = a.take<float>(T * 2 * HV, ok);
        m.gate = a.take<float>(T * HV, ok); m.beta = a.take<float>(T * HV, ok); m.hbuf = a.take<float>(T * C, ok);
        m.y = a.take<float>(T * ZV, ok); m.y_h = a.take<uint16_t>(T * ZV, ok);
        Alloc b;
        b.base = base; b.cap = region; b.owned = &m.owned;
        m.Kc = b.take<float>(T * 512, ok); m.Vc = b.take<float>(T * 512, ok); m.Qf = b.take<float>(T * 12288, ok);
        m.q = b.take<float>(T * ZV, ok); m.idx_raw = b.take<float>(T * 128, ok); m.q_idx = b.take<float>(T * 512, ok);
        m.attn = b.take<float>(T * ZV, ok); m.attn_h = b.take<uint16_t>(T * ZV, ok);
        m.sel_ids = b.take<int32_t>(T * (size_t) m.cap, ok);
        m.sel_scores = b.take<float>((size_t) m.sel_batch * (size_t) m.max_blocks, ok);
        m.attn_scratch = b.take<float>((size_t) m.attn_batch * strata::kernels::qsa_decode_attn_scratch_floats(m.cap, s), ok);
        Alloc c;
        c.base = base; c.cap = region; c.owned = &m.owned;
        m.logits = c.take<float>(T * m.g->n_expert, ok); m.w = c.take<float>(T * K, ok); m.ids = c.take<int32_t>(T * K, ok);
        m.slot_dev = c.take<int32_t>(T * K, ok); m.src_dev = c.take<int32_t>(T * K, ok);
        const MmqPlan& mp = mmq_plan();
        m.Xs = mp.fallback ? c.take<uint16_t>(T * K * N, ok) : nullptr;
        m.GU = c.take<float>(T * K * 1280, ok);
        m.Hh = mp.fallback ? c.take<uint16_t>(T * K * 640, ok) : nullptr;
        m.Dm = c.take<float>(T * K * N, ok);
        m.sgate = c.take<float>(T * 640, ok); m.sup = c.take<float>(T * 640, ok); m.sh_h = c.take<uint16_t>(T * 640, ok);
        m.shared = c.take<float>(T * N, ok); m.sg = c.take<float>(T, ok);
        if (mp.any) {
            m.Xq = c.take<uint8_t>(mmq::q8_bytes((int64_t) (T * K), N), ok);
            m.H = c.take<float>(T * K * 640, ok);
            m.Hq = c.take<uint8_t>(mmq::q8_bytes((int64_t) (T * K), 640), ok);
        }
        if (base == nullptr) ok = false;
    }
    for (int i = 0; i < DQ; ++i) { m.dq_gu[i] = o.take<uint16_t>(1280 * 2560, ok); m.dq_d[i] = o.take<uint16_t>(2560 * 640, ok); }
    if (mmq_plan().any) {
        const MmqPlan& mp = mmq_plan();
        m.ids_identity = o.take<int32_t>(T * K, ok);
        m.bounds_dev = o.take<int32_t>((size_t) (2 * (m.g->n_expert + m.g->n_expert / MMQ_GROUP + 2)), ok);
        m.grp_gu = o.take<uint8_t>(MMQ_GROUP * mp.gu_max + MMQ_TAIL, ok);
        m.grp_d = o.take<uint8_t>(MMQ_GROUP * mp.d_max + MMQ_TAIL, ok);
        // (written at every run's start, not here: when serving, these are live expert-cache slots until a request
        // lends them - a write now would corrupt a resident expert)
        if (!m.mmq_ctx) m.mmq_ctx = std::make_unique<mmq::Context>();
    }
    m.ring = ring_slots(T);
    for (int i = 0; i < m.ring; ++i) {
        m.stage_dev[i] = o.take<uint8_t>((size_t) MAXBLOB(), ok);
        m.stage_live[i] = false;                        // a new buffer: nothing of an earlier layout to wait for
    }
    m.ple_emb = o.take<float>(T * N, ok);
    m.ple_norm = o.take<float>((size_t) strata::kernels::NG_HC_DIM, ok);
    take_stage(o, ss, s, m.has_qsa, m.stage_pool, ok);
    m.T = (int64_t) T;
    return ok;
}

bool Prefill::relayout(int64_t chunk, const std::vector<Borrow>& borrow, std::string& err) {
    if (borrow.size() != parts_.size()) { err = "prefill: relayout needs one borrowed region per GPU"; return false; }
    for (size_t i = 0; i < parts_.size(); ++i) {
        Impl& m = *parts_[i];
        if (!m.borrowed || borrow[i].base == nullptr || chunk <= 0 || chunk > m.T_max) {
            err = "prefill: relayout needs borrowed buffers and a chunk of at most " + std::to_string(m.T_max);
            return false;
        }
        core::DeviceGuard dg(m.device);
        if (cudaStreamSynchronize(m.cs) != cudaSuccess || cudaStreamSynchronize(m.copy) != cudaSuccess) {
            err = "prefill: relayout: the stream failed";
            return false;
        }
        bool ok = true;
        Alloc o;
        o.base = (uint8_t*) borrow[i].base;
        o.cap = borrow[i].bytes;
        o.owned = &m.owned;
        uint16_t* gs = o.take<uint16_t>((size_t) GEMM_SCRATCH, ok);
        void* ws = o.take<uint8_t>(GEMM_WS, ok);
        if (ok) m.gemm.rebind(gs, GEMM_SCRATCH, ws, GEMM_WS);
        if (!ok || !carve(m, (size_t) chunk, &o)) {
            err = "prefill: device buffers for a chunk of " + std::to_string(chunk) + " tokens do not fit on GPU " +
                  std::to_string(m.device);
            return false;
        }
    }
    return true;
}

int64_t Prefill::chunk() const { return parts_.front()->T; }
void Prefill::set_pinned_share(double share) { g_pinned_share = share; }
double Prefill::pinned_share() { return g_pinned_share; }

uint64_t Prefill::bytes_needed(const core::ModelGeometry& g, const core::SessionState& ss, int64_t chunk, int stage) {
    // the same allocation sequence as `init`, counted
    const size_t T = (size_t) chunk;
    bool ok = true;
    Alloc o;
    o.count_only = true;
    o.take<uint16_t>((size_t) GEMM_SCRATCH, ok);
    o.take<uint8_t>(GEMM_WS, ok);
    auto f = [&](size_t n) { o.take<float>(n, ok); };
    f(T * N); f(T * D); f(T * HC); o.take<uint16_t>(T * D, ok); f(T * LR); o.take<uint16_t>(T * LR, ok);
    f(T * D); f(T * HC); f(T * N); o.take<uint16_t>(T * N, ok); o.take<uint16_t>(T * N, ok); f(T * N);
    o.take<int32_t>(T * strata::kernels::kStepCount, ok);
    strata::kernels::QsaShapes s = strata::kernels::qsa_real_shapes();
    s.n_head = g.n_head; s.n_head_kv = g.n_head_kv; s.head_dim = g.head_dim; s.idx_n_head = g.idx_q_heads;
    s.idx_dim = g.idx_key_dim;
    const int64_t cap = strata::kernels::qsa_selection_width(strata::kernels::kTopkMaxCells, s);
    const int64_t max_blocks = ss.qsa_states[0].max_cells / s.idx_block + 2;
    o.take<uint8_t>((size_t) std::max({gdn_set_bytes(T), qsa_set_bytes(T, cap, max_blocks, 256, 32, s),
                                       moe_set_bytes(T, g.n_expert)}), ok);
    for (int i = 0; i < DQ; ++i) { o.take<uint16_t>(1280 * 2560, ok); o.take<uint16_t>(2560 * 640, ok); }
    if (mmq_plan().any) {
        const MmqPlan& mp = mmq_plan();
        o.take<int32_t>(T * K, ok);
        o.take<int32_t>((size_t) (2 * (g.n_expert + g.n_expert / MMQ_GROUP + 2)), ok);
        o.take<uint8_t>(MMQ_GROUP * mp.gu_max + MMQ_TAIL, ok);
        o.take<uint8_t>(MMQ_GROUP * mp.d_max + MMQ_TAIL, ok);
    }
    for (int i = 0; i < ring_slots(T); ++i) o.take<uint8_t>((size_t) MAXBLOB(), ok);
    f(T * N);
    f((size_t) strata::kernels::NG_HC_DIM);
    bool has_qsa = false;
    for (int64_t l = core::placement().first(stage); l < core::placement().end(stage); ++l)
        has_qsa = has_qsa || core::is_qsa_layer(g, l);
    strata::kernels::KvHostPools pool;
    take_stage(o, ss, s, has_qsa, pool, ok);
    return o.used + (8u << 20);   // alignment slack
}

namespace {

const core::WeightRef* need(const core::LayerView& v, const char* suffix, std::string& err) {
    const core::WeightRef* r = v.get(suffix);
    if (!r) err = v.name(suffix) + " is missing";
    return r;
}
bool native_proj(Gemm& gm, const core::WeightRef* w, const uint16_t* X, float* Y, int64_t T, const std::string& name,
                 std::string& err, int64_t ldy = 0) {
    if (!w->native_data) { err = "prefill: " + name + " has no native GGUF blocks (run with --native)"; return false; }
    gm.native(X, w->native_type, w->native_data, Y, T, w->ne1, w->ne0, ldy);
    return true;
}
bool bf16_proj(Gemm& gm, const core::WeightRef* w, const uint16_t* X, float* Y, int64_t T, const std::string& name,
               std::string& err, int64_t ldy = 0) {
    if (w->kind != core::WeightKind::Bf16InF32 || !w->data) { err = "prefill: " + name + " is not a resident BF16 tensor"; return false; }
    gm.bf16(X, (const uint16_t*) w->data, Y, T, w->ne1 > 0 ? w->ne1 : 1, w->ne0, ldy);
    return true;
}

}  // namespace

namespace {
// STRATA_PREFILL_TIMING=1: the prompt path's GPU time by phase.  Events are recorded on the compute stream in order;
// the time between two consecutive marks is charged to the phase of the first, so a gap where the GPU waits (for the
// host's expert grouping, or for an expert's copy) lands on the phase that was waiting.  Events are reused: the marks
// are folded at every MoE layer's host sync, after which all of them have completed.
enum PfPhase { kPfStart, kPfHc, kPfGdn, kPfGdnConv, kPfGdnRec, kPfGdnOut, kPfQsa, kPfQsaIdx, kPfQsaSel, kPfQsaAttn,
               kPfRouter, kPfHostGroup, kPfGather, kPfWaitCopy, kPfDequant, kPfGemmGU, kPfGemmD, kPfCombine, kPfPle,
               kPfCount };
const char* const kPfNames[kPfCount] = {"embed+steps", "hc read", "gdn proj in", "gdn conv+gates", "gdn recurrence",
                                        "gdn proj out", "qsa proj", "qsa indexer", "qsa select", "qsa attn",
                                        "router+shared", "host grouping", "gather", "wait copy", "dequant",
                                        "gemm gate/up", "gemm down", "combine", "ple"};
struct PfTimer {
    bool on = std::getenv("STRATA_PREFILL_TIMING") != nullptr;
    std::vector<cudaEvent_t> ev;
    std::vector<int> ph;
    size_t used = 0;
    double ms[kPfCount] = {};
    void mark(int phase, cudaStream_t s) {
        if (!on) return;
        if (used == ev.size()) {
            cudaEvent_t e = nullptr;
            cudaEventCreate(&e);
            ev.push_back(e);
            ph.push_back(0);
        }
        ph[used] = phase;
        cudaEventRecord(ev[used], s);
        ++used;
    }
    // every recorded mark has completed (the stream was synchronized): charge the gaps, keep the last mark
    void fold() {
        if (!on || used < 2) return;
        for (size_t i = 0; i + 1 < used; ++i) {
            float t = 0.0f;
            if (cudaEventElapsedTime(&t, ev[i], ev[i + 1]) == cudaSuccess) ms[ph[i]] += t;
        }
        std::swap(ev[0], ev[used - 1]);
        std::swap(ph[0], ph[used - 1]);
        used = 1;
    }
    ~PfTimer() {
        for (cudaEvent_t e : ev) cudaEventDestroy(e);
    }
};
}  // namespace

bool Prefill::run(const int64_t* tokens, int64_t n, int64_t pos0, std::string& err) {
    Impl& m0 = *parts_.front();   // the first stage: the embeddings
    Impl& ml = *parts_.back();    // the last stage: the residual the callback gets
    Impl& mp = *parts_[(size_t) core::placement().stage_of(1)];   // layer 1's stage: the PLE
    const core::ModelGeometry& g = *m0.g;
    core::SessionState& ss = *m0.ss;
    const auto t_start = Clock::now();
    strata::kernels::QsaShapes s = strata::kernels::qsa_real_shapes();
    s.n_head = g.n_head; s.n_head_kv = g.n_head_kv; s.head_dim = g.head_dim; s.idx_n_head = g.idx_q_heads;
    s.idx_dim = g.idx_key_dim;
    const uint64_t gdn_floats = core::gdn_state_floats(g);
    int32_t prev[2] = {ss.ple_prev[0], ss.ple_prev[1]};
    std::vector<PfTimer> timers(parts_.size());   // one per stage: an event belongs to one GPU
    std::vector<PrefillStats> part_stats(parts_.size());
    // the MMQ row table lives in the borrowed cache slots, which the refill after a prompt overwrites with experts:
    // write it again for every prompt (a layout is reused as long as the chunk and the slots are the same)
    for (auto& part : parts_) {
        core::DeviceGuard dg(part->device);
        if (part->ids_identity != nullptr) mmq::iota(part->ids_identity, part->T * K, part->cs);
    }
    // The PLE rows of a chunk are read from the model file on the host (an SSD read per missed row): the chunk
    // after this one is read on a thread while the GPU runs this one, into the other of two buffers.  The rows
    // depend only on the tokens (the two before a position name its n-grams), so this is the same data.
    const bool ple_on = ss.ple.ready();
    // the PLE block batched over the chunk: the pinned postops and a BF16 or GGUF-native key (else token by token);
    // STRATA_PLE_BATCH=0 keeps the per-token block (the A/B)
    static const bool ple_batch_env = [] {
        const char* v = std::getenv("STRATA_PLE_BATCH");
        return v == nullptr || std::atoi(v) != 0;
    }();
    const bool ple_batch = ple_on && ple_batch_env && strata::kernels::ple_native_postops_enabled() &&
                           (ss.ple.w.key_bf16 != nullptr || ss.ple.w.key_native_data != nullptr) &&
                           mp.region_bytes / ((uint64_t) (3 * strata::kernels::NG_HC_DIM + N + 4) * 4 + (uint64_t) N * 2 + 4096) >= 64;
    const int32_t prev0[2] = {prev[0], prev[1]};
    auto ple_gather = [&m = mp, &ss, tokens, n, prev0](int64_t c0, int buf, std::string& e) -> bool {
        const int64_t T = std::min(m.T, n - c0);
        auto at = [&](int64_t i) { return i < 2 ? prev0[i] : (int32_t) tokens[i - 2]; };   // prev0, then the tokens
        int32_t pv[2] = {at(c0), at(c0 + 1)};
        for (int64_t t = 0; t < T; ++t) {
            const int32_t tok = (int32_t) tokens[c0 + t];
            strata::kernels::ngram_rows(&tok, pv, 1, ss.ple.consts,
                                        m.ple_rows[buf].data() + t * strata::kernels::PLE_N_HEADS);
            pv[0] = pv[1];
            pv[1] = tok;
        }
        return ss.ple.table->gather_batch(m.ple_rows[buf].data(), (size_t) T, m.ple_emb_host[buf], e);
    };

    // THE STAGES AS A PIPELINE: each stage runs every chunk of the prompt through its layers on its own host thread,
    // so the first GPU reads chunk k+1 while the next one reads chunk k.  A stage takes a chunk's residual once the
    // stage before it has finished that chunk (`done`), and the stage before writes its residual again only once
    // it has been taken (`handed`).  At a chunk end `hold_at` names, the first stage waits for the last one, so the
    // callback there sees every layer at the same position.
    const size_t n_parts = parts_.size();
    std::mutex mu;
    std::condition_variable cv;
    std::vector<int64_t> done(n_parts, 0), handed(n_parts, 0);   // chunks finished / taken over, per stage
    bool aborted = false;
    std::string first_err;
    auto fail = [&](const std::string& e) {
        std::lock_guard<std::mutex> lk(mu);
        if (!aborted) { aborted = true; first_err = e; }
        cv.notify_all();
    };
    auto wait_until = [&](auto pred) {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&] { return aborted || pred(); });
        return !aborted;
    };
    auto signal = [&](std::vector<int64_t>& v, size_t i, int64_t k) {
        std::lock_guard<std::mutex> lk(mu);
        v[i] = k;
        cv.notify_all();
    };

    auto run_part = [&](size_t pi, std::string& err) -> bool {
        Impl& m = *parts_[pi];
        core::DeviceGuard dg(m.device);
        const cudaStream_t cs = m.cs;
        PfTimer& pt = timers[pi];
        PrefillStats& pst = part_stats[pi];
        std::string ple_next_err;
        std::future<bool> ple_next;             // declared after everything it reads: an early return waits for it
        int ple_buf = 0;
        auto ple_start = [&](int64_t c, int b) {
            ple_next = std::async(std::launch::async, [&ple_gather, &ple_next_err, c, b] { return ple_gather(c, b, ple_next_err); });
        };
        // a chunk's PLE rows are read on a thread from the chunk before (the first chunk's from the start, behind its
        // embedding and layer 0); layer 1 takes them: the upload, then the next chunk's read
        auto ple_take = [&](int64_t c0, int64_t T, std::string& err) -> bool {
            const auto tp = Clock::now();
            if (!ple_next.get()) { err = ple_next_err; return false; }
            cudaMemcpyAsync(m.ple_emb, m.ple_emb_host[ple_buf], (size_t) T * N * 4, cudaMemcpyHostToDevice, m.cs);
            cudaEventRecord(m.ple_copied[ple_buf], m.cs);
            if (c0 + m.T < n) {
                cudaEventSynchronize(m.ple_copied[ple_buf ^ 1]);   // the other buffer's upload (a chunk ago) is done
                ple_start(c0 + m.T, ple_buf ^ 1);
            }
            ple_buf ^= 1;
            pst.ms_ple += ms_since(tp);
            return true;
        };
        if (ple_on && &m == &mp && n > 0) ple_start(0, 0);
        bool hold = false;
        for (int64_t k = 0, c0 = 0; c0 < n; ++k, c0 += m0.T) {
            const int64_t T = std::min(m0.T, n - c0), p0 = pos0 + c0;
            if (should_stop && should_stop()) { err = "cancelled"; return false; }
            if (pi == 0) {
                if (std::getenv("STRATA_TRACE")) { std::fprintf(stderr, "strata trace: prompt chunk %lld of %lld\n", (long long) c0, (long long) n); std::fflush(stderr); }
                if (hold && !wait_until([&] { return done[n_parts - 1] >= k; })) return false;
            }
            // the next stage has taken this stage's previous residual
            if (pi + 1 < n_parts && k > 0) {
                if (!wait_until([&] { return handed[pi + 1] >= k; })) return false;
                cudaStreamWaitEvent(cs, parts_[pi + 1]->handed, 0);
            }
            ++pst.chunks;
            pt.mark(kPfStart, cs);
            if (pi == 0) {
                // ---- embeddings, broadcast to the four streams: the chunk's rows in one gather (the per-token
                // path's arithmetic); a chunk with picture rows, or a token outside the table, takes the per-token path
                const core::NativeEmbed* nemb = core::native_embed();
                const core::WeightRef* wemb = nemb ? nullptr : m.wt->find("token_embd.weight");
                bool batched = nemb != nullptr ||
                               (wemb != nullptr && !wemb->codebook_iq4nl && wemb->ne0 == g.n_embd && wemb->group_elems > 0 &&
                                (wemb->code_bits == 2 || wemb->code_bits == 4 || wemb->code_bits == 8));
                for (int64_t t = 0; batched && t < T; ++t) {
                    const int64_t tok = tokens[c0 + t];
                    if ((embd_rows && embd_rows[p0 + t]) || tok < 0 || (wemb && tok >= wemb->ne1)) batched = false;
                    else m.tok_host[(size_t) t] = (int32_t) tok;
                }
                if (batched) {
                    if (cudaMemcpyAsync(m.tok_dev, m.tok_host.data(), (size_t) T * sizeof(int32_t), cudaMemcpyHostToDevice,
                                        m.cs) != cudaSuccess) {
                        err = "prefill: the token id upload failed";
                        return false;
                    }
                    if (nemb) {
                        nemb->gather_dev(m.tok_dev, T, m.emb, m.cs);
                    } else {
                        const auto* codes = (const uint8_t*) wemb->data;
                        const auto* scales = (const float*) (codes + wemb->codes_bytes);
                        const auto* offsets = wemb->has_offset ? (const float*) (codes + wemb->codes_bytes + wemb->scales_bytes)
                                                               : nullptr;
                        strata::kernels::embedding_gather_dev(codes, scales, offsets, m.tok_dev, (int) T, wemb->ne0,
                                                              wemb->code_bits, wemb->code_bias, wemb->group_elems,
                                                              (uint64_t) (wemb->ne0 / (8 / wemb->code_bits)),
                                                              (uint64_t) (wemb->ne0 / wemb->group_elems), m.emb, m.cs);
                    }
                }
                for (int64_t t = 0; !batched && t < T; ++t) {
                    const float* row = embd_rows ? embd_rows[p0 + t] : nullptr;
                    if (row) {
                        if (cudaMemcpyAsync(m.emb + t * N, row, (size_t) N * 4, cudaMemcpyHostToDevice, m.cs) != cudaSuccess) {
                            err = "prefill: the image embedding upload failed";
                            return false;
                        }
                    } else if (!core::embed_row(*m.wt, g, tokens[c0 + t], m.emb + t * N, m.cs, err)) {
                        return false;
                    }
                }
                gr_broadcast(m.emb, m.R, T, m.cs);
                for (int64_t t = 0; t < T; ++t) { prev[0] = prev[1]; prev[1] = (int32_t) tokens[c0 + t]; }
            } else {
                // the chunk's residual from the previous stage's GPU (staged through host memory without peer access)
                if (!wait_until([&] { return done[pi - 1] > k; })) return false;
                Impl& mq = *parts_[pi - 1];
                cudaStreamWaitEvent(cs, mq.done, 0);
                cudaMemcpyPeerAsync(m.R, m.device, mq.R, mq.device, (size_t) T * D * 4, cs);
                cudaEventRecord(m.handed, cs);
                signal(handed, pi, k + 1);
            }
            // ---- the QSA step records of every position in the chunk
            for (int64_t t = 0; t < T; ++t) strata::kernels::qsa_step_fill(m.steps_host.data() + t * strata::kernels::kStepCount, p0 + t, s);
            cudaMemcpyAsync(m.steps_dev, m.steps_host.data(), (size_t) T * strata::kernels::kStepCount * 4,
                            cudaMemcpyHostToDevice, m.cs);
            int64_t qsa_index = m.qsa0, gdn_index = m.gdn0;
            // step 3: this chunk's stream - every non-resident expert of every layer, layer by layer in id order (entry
            // k lands in ring slot k % ring); a copy is issued once the entry `ring` before it is consumed (its slot's
            // `used` event recorded), so the copy stream never waits on an event that is not queued yet
            const strata::kernels::cpu::ExpertLayout& lay0 = strata::kernels::cpu::expert_layout();
            const bool stream_all = m.ring > STAGE && T >= STREAM_ALL_MIN && m.src != nullptr;
            struct StreamEntry { int32_t l, e; const uint8_t* blob; int job; };
            std::vector<StreamEntry> seq;
            std::vector<size_t> seq_start;
            size_t issued = 0, consumed = 0;
            // The first GPU's first chunk, while the next GPU waits for it: that GPU (`helper`) computes the streamed
            // experts - their blobs cross its x16 link to its own ring, only their activations and outputs cross this
            // GPU's link - with its idle buffers, on its stream ahead of its own first chunk.  It streams every
            // non-resident expert, as the stream-all walk does, whatever the chunk.  Every layer MMQ, every streamed blob
            // pinned (the helper DMAs them itself).
            Impl* helper = nullptr;
            if (pi == 0 && k == 0 && n_parts > 1 && m.src != nullptr) {
                bool fit = mmq_plan().any;
                for (int64_t l = m.l0; fit && l < m.l1; ++l) {
                    fit = mmq_plan().layer[(size_t) l] != 0;
                    for (int32_t e = 0; fit && e < m.g->n_expert; ++e)
                        if (!m.resident(l, e))
                            fit = m.src->blob(l, e) != nullptr && m.src->pinned(l, e);
                }
                if (fit) helper = parts_[1].get();
            }
            std::vector<StreamEntry> hseq;              // the helper's stream: what `seq` would hold
            size_t h_issued = 0, h_consumed = 0;
            if (stream_all || helper) {
                seq_start.resize((size_t) g.n_layers + 1);
                std::vector<Stager::Job> js;
                std::vector<StreamEntry>& sq = helper ? hseq : seq;
                for (int64_t l = m.l0; l < m.l1; ++l) {
                    seq_start[(size_t) l] = sq.size();
                    for (int32_t e = 0; e < m.g->n_expert; ++e) {
                        if (m.resident(l, e)) continue;
                        const uint8_t* b = m.src->blob(l, e);
                        if (!b) { err = "prefill: expert source has no blob"; return false; }
                        int job = -1;
                        if (!m.src->pinned(l, e)) {
                            job = (int) js.size();
                            js.push_back({b, (size_t) lay0.blob_bytes(l)});
                        }
                        sq.push_back({(int32_t) l, e, b, job});
                    }
                }
                seq_start[(size_t) m.l1] = sq.size();
                if (stream_all) m.stager->start(std::move(js));
            }
            // the helper's copies: entry k lands in its ring slot k % ring, issued once the entry `ring` before it is
            // consumed (the slot's `used` event recorded on the helper's stream)
            auto h_issue_until = [&](size_t limit) {
                Impl& h = *helper;
                core::DeviceGuard dgh(h.device);
                limit = std::min(limit, hseq.size());
                while (h_issued < limit) {
                    const StreamEntry& en = hseq[h_issued];
                    const int sl = (int) (h_issued % (size_t) h.ring);
                    if (h.stage_live[sl]) cudaStreamWaitEvent(h.copy, h.used[sl], 0);
                    cudaMemcpyAsync(h.stage_dev[sl], en.blob, (size_t) lay0.blob_bytes(en.l), cudaMemcpyHostToDevice, h.copy);
                    cudaEventRecord(h.copied[sl], h.copy);
                    h.stage_live[sl] = true;
                    ++pst.experts_helper;
                    ++h_issued;
                }
            };
            if (helper) h_issue_until((size_t) helper->ring);
            struct StagerDone {
                Stager* st;
                ~StagerDone() { if (st) st->finish(); }
            } chunk_stager_done{stream_all ? m.stager.get() : nullptr};
            auto issue_until = [&](size_t limit) {
                limit = std::min(limit, seq.size());
                while (issued < limit) {
                    const StreamEntry& en = seq[issued];
                    const int sl = (int) (issued % (size_t) m.ring);
                    const auto th = Clock::now();
                    const size_t bytes = (size_t) lay0.blob_bytes(en.l);
                    if (m.stage_live[sl]) cudaStreamWaitEvent(m.copy, m.used[sl], 0);
                    if (en.job < 0) {
                        cudaMemcpyAsync(m.stage_dev[sl], en.blob, bytes, cudaMemcpyHostToDevice, m.copy);
                        ++pst.experts_dma;
                    } else {
                        const uint8_t* hb = m.stager->wait(en.job);
                        cudaMemcpyAsync(m.stage_dev[sl], hb, bytes, cudaMemcpyHostToDevice, m.copy);
                        m.stager->issued_one(en.job, m.copy);
                    }
                    cudaEventRecord(m.copied[sl], m.copy);
                    m.stage_live[sl] = true;
                    pst.ms_experts_host += ms_since(th);
                    ++pst.experts_streamed;
                    ++issued;
                }
            };
            if (stream_all) issue_until((size_t) m.ring);   // the stage's first experts, behind the embedding
            for (int64_t l = m.l0; l < m.l1; ++l) {
                core::progress_beat();   // the serve watchdog: a prompt chunk of 8192 tokens is still moving
                const core::LayerView v(*m.wt, l);
                // ---- the PLE block at layer 1, token by token (its conv reads the previous tokens' rows)
                if (l == 1 && ple_on && !ple_take(c0, T, err)) return false;
                if (l == 1 && ple_on && ple_batch) {
                    // the whole chunk at once, in sub-batches carved from the idle scratch region: the key and value
                    // projections as GEMMs (a token at a time they re-read ~52 MB of BF16 key per token on the IQ
                    // files), the rest with the per-token kernels' arithmetic (native_ple_postops_batch)
                    pt.mark(kPfPle, cs);
                    const auto tp = Clock::now();
                    const strata::kernels::PleWeights& pw = ss.ple.w;
                    constexpr int64_t HD = strata::kernels::NG_HC_DIM;
                    const uint64_t per_token = (uint64_t) (3 * HD + N + 4) * 4 + (uint64_t) N * 2 + 4096;
                    const int64_t SB = std::min<int64_t>(T, (int64_t) (m.region_bytes / per_token));
                    for (int64_t s0 = 0; s0 < T; s0 += SB) {
                        const int64_t nb = std::min(SB, T - s0);
                        uint8_t* q = m.region;
                        auto carve_f = [&](size_t n) { float* p = (float*) q; q += (n * 4 + 255) & ~(size_t) 255; return p; };
                        float* key = carve_f((size_t) nb * HD);
                        float* qn = carve_f((size_t) nb * HD);
                        float* gated = carve_f((size_t) nb * HD);
                        float* val = carve_f((size_t) nb * N);
                        float* gate = carve_f((size_t) nb * 4);
                        uint16_t* e16 = (uint16_t*) carve_f((size_t) nb * N / 2);
                        const float* emb = m.ple_emb + s0 * N;
                        if (pw.key_bf16 != nullptr) {
                            to_bf16(emb, e16, nb * N, m.cs);
                            m.gemm.bf16(e16, pw.key_bf16, key, nb, HD, N);
                        } else {
                            to_f16(emb, e16, nb * N, m.cs);
                            m.gemm.native(e16, pw.key_native_type, pw.key_native_data, key, nb, HD, N);
                            to_bf16(emb, e16, nb * N, m.cs);
                        }
                        m.gemm.bf16(e16, pw.value_bf16, val, nb, N, N);
                        try {
                            strata::kernels::native_ple_postops_batch(key, m.R + s0 * D, val, ss.ple.hist, pw, qn, gated,
                                                                      gate, (int) nb, m.cs);
                        } catch (const std::exception& e) { err = std::string("prefill PLE: ") + e.what(); return false; }
                    }
                    pst.ms_ple += ms_since(tp);
                } else if (l == 1 && ple_on) {
                    pt.mark(kPfPle, cs);
                    const auto tp = Clock::now();
                    for (int64_t t = 0; t < T; ++t) {
                        strata::kernels::PleOut po;
                        po.normalized = m.ple_norm;
                        po.result = m.R + t * D;
                        try {
                            strata::kernels::ple_block(m.ple_emb + t * N, m.R + t * D, ss.ple.hist, ss.ple.w, po,
                                                       ss.ple.scratch, m.cs);
                        } catch (const std::exception& e) { err = std::string("prefill PLE: ") + e.what(); return false; }
                        strata::kernels::ple_history_advance(ss.ple.hist, m.ple_norm, m.cs);
                    }
                    pst.ms_ple += ms_since(tp);
                }
                for (int half = 0; half < 2; ++half) {
                    // ---- the hyper-connection read of this half
                    const char* pre = half == 0 ? "hc_attn_" : "hc_ffn_";
                    const std::string sn = std::string(pre) + "norm.weight", sd = std::string(pre) + "down.weight",
                                      su = std::string(pre) + "up.weight", si = std::string(pre) + "inject.weight";
                    const core::WeightRef *wn = need(v, sn.c_str(), err), *wd = need(v, sd.c_str(), err),
                                          *wu = need(v, su.c_str(), err), *wi = need(v, si.c_str(), err);
                    if (!wn || !wd || !wu || !wi) return false;
                    pt.mark(kPfHc, cs);
                    gr_norm(m.R, (const float*) wn->data, EPS, m.row_rs, m.xn16, T, m.cs);
                    if (!bf16_proj(m.gemm, wd, m.xn16, m.lo, T, sd, err)) return false;
                    gr_silu(m.lo, m.lo16, T, m.cs);
                    if (!bf16_proj(m.gemm, wu, m.lo16, m.gated, T, su, err)) return false;
                    if (!bf16_proj(m.gemm, wi, m.xn16, m.inj, T, si, err)) return false;
                    gr_mix(m.R, (const float*) wn->data, m.row_rs, m.gated, m.mixed, m.mixed_bf, T, m.cs, m.mixed_h);

                    if (half == 0 && !core::is_qsa_layer(g, l)) {
                        // ======================= GDN =======================
                        const core::WeightRef *wqkv = need(v, "attn_qkv.weight", err), *wg = need(v, "attn_gate.weight", err),
                                              *wo = need(v, "ssm_out.weight", err), *wa = need(v, "ssm_alpha.weight", err),
                                              *wb = need(v, "ssm_beta.weight", err), *wc = need(v, "ssm_conv1d.weight", err),
                                              *wnm = need(v, "ssm_norm.weight", err), *wdt = need(v, "ssm_dt.bias", err),
                                              *wsa = need(v, "ssm_a", err);
                        if (!wqkv || !wg || !wo || !wa || !wb || !wc || !wnm || !wdt || !wsa) return false;
                        pt.mark(kPfGdn, cs);
                        float* state = ss.gdn_states[(size_t) gdn_index];
                        float* conv = state + (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size;
                        if (!native_proj(m.gemm, wqkv, m.mixed_h, m.qkv, T, v.name("attn_qkv.weight"), err)) return false;
                        if (!native_proj(m.gemm, wg, m.mixed_h, m.z, T, v.name("attn_gate.weight"), err)) return false;
                        if (!bf16_proj(m.gemm, wa, m.mixed_bf, m.ab, T, v.name("ssm_alpha.weight"), err, 2 * HV)) return false;
                        if (!bf16_proj(m.gemm, wb, m.mixed_bf, m.ab + HV, T, v.name("ssm_beta.weight"), err, 2 * HV)) return false;
                        pt.mark(kPfGdnConv, cs);
                        gdn_gates(m.ab, (const float*) wdt->data, (const float*) wsa->data, m.gate, m.beta, T, m.cs);
                        gdn_conv(conv, m.qkv, (const float*) wc->data, m.hbuf, T, EPS, m.cs);
                        pt.mark(kPfGdnRec, cs);
                        gdn_recurrence(state, m.hbuf, m.gate, m.beta, m.z, (const float*) wnm->data, EPS, m.y, m.y_h, T, m.cs);
                        pt.mark(kPfGdnOut, cs);
                        if (!native_proj(m.gemm, wo, m.y_h, m.bo, T, v.name("ssm_out.weight"), err)) return false;
                        ++gdn_index;
                    } else if (half == 0) {
                        // ======================= QSA =======================
                        const core::QsaState& st = ss.qsa_states[qsa_index];
                        const core::WeightRef *wq = need(v, "attn_q.weight", err), *wk = need(v, "attn_k.weight", err),
                                              *wv = need(v, "attn_v.weight", err), *wo = need(v, "attn_output.weight", err),
                                              *wik = need(v, "indexer.k_proj.weight", err),
                                              *wiq = need(v, "indexer.q_proj.weight", err),
                                              *wqn = need(v, "attn_q_norm.weight", err), *wkn = need(v, "attn_k_norm.weight", err),
                                              *wiqn = need(v, "indexer.q_norm.weight", err),
                                              *wikn = need(v, "indexer.k_norm.weight", err);
                        if (!wq || !wk || !wv || !wo || !wik || !wiq || !wqn || !wkn || !wiqn || !wikn) return false;
                        pt.mark(kPfQsa, cs);
                        if (!native_proj(m.gemm, wk, m.mixed_h, m.Kc, T, v.name("attn_k.weight"), err)) return false;
                        if (!native_proj(m.gemm, wv, m.mixed_h, m.Vc, T, v.name("attn_v.weight"), err)) return false;
                        if (!native_proj(m.gemm, wq, m.mixed_h, m.Qf, T, v.name("attn_q.weight"), err)) return false;
                        if (!bf16_proj(m.gemm, wik, m.mixed_bf, m.idx_raw, T, v.name("indexer.k_proj.weight"), err)) return false;
                        if (!bf16_proj(m.gemm, wiq, m.mixed_bf, m.q_idx, T, v.name("indexer.q_proj.weight"), err)) return false;
                        rms_rows(m.Kc, (const float*) wkn->data, T * 2, 256, 256, EPS, m.cs);
                        rope(m.Kc, T, 2, 256, 512, p0, (float) strata::kernels::qsa_freq_base(), m.cs);
                        // KV streaming: this layer's cells [0, p0) come in from the host copy to the staging pool, and the
                        // chunk's cells go to the host copy, the staging pool, and the VRAM slots of resident blocks
                        const bool staged = st.kv_mode == 1;
                        if (staged)
                            strata::kernels::kv_stage_from_host(pools_of(m.stage_pool, m.ident_table), st.host,
                                                                core::qsa_kv_format(st),
                                                                (p0 + s.page_size - 1) / s.page_size, s, m.cs);
                        if (st.kv_q4) {   // Q4_0 KV (kv_q4.hpp): rotated K and V, the queries below too, the output back
                            strata::kernels::fwht256_inplace_cuda(m.Kc, T * 2, m.cs);
                            strata::kernels::fwht256_inplace_cuda(m.Vc, T * 2, m.cs);
                            strata::kernels::kv_append_q4(st.k_q4, st.v_q4, st.page_table, p0, T, m.Kc, m.Vc, s, m.cs,
                                                          &st.host, staged ? &m.stage_pool : nullptr);
                        } else {
                            kv_append(m.Kc, m.Vc, T, p0, st.page_table, s.page_size, st.kv_int8 ? nullptr : st.k_pool,
                                      st.kv_int8 ? nullptr : st.v_pool, st.k_q, st.v_q, st.k_scale, st.v_scale, m.cs,
                                      &st.host, staged ? &m.stage_pool : nullptr);
                        }
                        split_q(m.Qf, m.q, T, m.cs);
                        rms_rows(m.q, (const float*) wqn->data, T * 24, 256, 256, EPS, m.cs);
                        rope(m.q, T, 24, 256, 6144, p0, (float) strata::kernels::qsa_freq_base(), m.cs);
                        if (st.kv_q4) strata::kernels::fwht256_inplace_cuda(m.q, T * 24, m.cs);
                        rms_rows(m.q_idx, (const float*) wiqn->data, T * 4, 128, 128, EPS, m.cs);
                        rope(m.q_idx, T, 4, 128, 512, p0, (float) strata::kernels::qsa_freq_base(), m.cs);
                        // the indexer appends the chunk (the end state of token-by-token appends); then scores + selection
                        // for many queries at once: a query reads completed blocks (final once completed) and `dead` for
                        // its own tail block
                        const strata::kernels::QsaIndexerBuffers ib{st.idx_tail, st.idx_dead, st.idx_pooled, st.idx_block_pos};
                        pt.mark(kPfQsaIdx, cs);
                        try {
                            strata::kernels::native_qsa_indexer_append_batch(m.idx_raw, T, p0, 0, (const float*) wikn->data, EPS,
                                                                             ib, s, st.max_cells,
                                                                             (float) strata::kernels::qsa_freq_base(), m.cs);
                        } catch (const std::exception& e) { err = std::string("prefill indexer: ") + e.what(); return false; }
                        pt.mark(kPfQsaSel, cs);
                        for (int64_t t0 = 0; t0 < T; t0 += m.sel_batch) {
                            const int64_t nb = std::min(m.sel_batch, T - t0);
                            const int32_t* steps0 = m.steps_dev + t0 * strata::kernels::kStepCount;
                            // the grid reaches the batch's last query's n_bid (they rise with the position)
                            const int64_t active = (int64_t) m.steps_host[(size_t) ((t0 + nb - 1) * strata::kernels::kStepCount +
                                                                                    strata::kernels::kStepNBid)] + 1;
                            strata::kernels::qsa_block_scores_tc(st.idx_pooled, st.idx_dead, m.q_idx + t0 * 512, steps0, nb,
                                                                 m.max_blocks, s, m.sel_scores, m.cs, active);
                            strata::kernels::qsa_block_topk(m.sel_scores, steps0, nb, m.max_blocks, m.cap, s,
                                                            m.sel_ids + t0 * m.cap, m.cs, active);
                        }
                        // STRATA_IDX_FP16_CHECK: would FP16 pooled indexer keys select the same cells? (the KV-streaming
                        // design's last question). Every query is selected again from the pooled keys and `dead` rounded
                        // to fp16 (exactly what an fp16 store reads back); the agreement with the fp32 selection is
                        // printed cumulatively after each chunk's last QSA layer. Debug: syncs per layer.
                        if (static const bool f16chk = std::getenv("STRATA_IDX_FP16_CHECK") != nullptr; f16chk) {
                            static float *pooled16 = nullptr, *dead16 = nullptr;
                            static int32_t* ids16 = nullptr;
                            static double shared = 0, cells = 0;
                            static long long queries = 0, same = 0, sel_queries = 0;
                            const int64_t rows = st.idx_pooled_rows;
                            if (pooled16 == nullptr &&
                                (cudaMalloc((void**) &pooled16, (size_t) rows * s.idx_dim * 4) != cudaSuccess ||
                                 cudaMalloc((void**) &dead16, (size_t) s.idx_dim * 4) != cudaSuccess ||
                                 cudaMalloc((void**) &ids16, (size_t) (m.T * m.cap) * 4) != cudaSuccess)) {
                                err = "STRATA_IDX_FP16_CHECK: no room for its buffers";
                                return false;
                            }
                            round_f16(st.idx_pooled, pooled16, rows * s.idx_dim, m.cs);
                            round_f16(st.idx_dead, dead16, s.idx_dim, m.cs);
                            for (int64_t t0 = 0; t0 < T; t0 += m.sel_batch) {
                                const int64_t nb = std::min(m.sel_batch, T - t0);
                                const int32_t* steps0 = m.steps_dev + t0 * strata::kernels::kStepCount;
                                strata::kernels::qsa_block_scores_tc(pooled16, dead16, m.q_idx + t0 * 512, steps0, nb,
                                                                     m.max_blocks, s, m.sel_scores, m.cs, m.max_blocks);
                                strata::kernels::qsa_block_topk(m.sel_scores, steps0, nb, m.max_blocks, m.cap, s,
                                                                ids16 + t0 * m.cap, m.cs);
                            }
                            std::vector<int32_t> a((size_t) (T * m.cap)), b((size_t) (T * m.cap));
                            cudaMemcpyAsync(a.data(), m.sel_ids, a.size() * 4, cudaMemcpyDeviceToHost, m.cs);
                            cudaMemcpyAsync(b.data(), ids16, b.size() * 4, cudaMemcpyDeviceToHost, m.cs);
                            cudaStreamSynchronize(m.cs);
                            for (int64_t t = 0; t < T; ++t) {
                                const int64_t w = m.steps_host[(size_t) (t * strata::kernels::kStepCount + strata::kernels::kStepWidth)];
                                const int32_t *x = a.data() + t * m.cap, *y = b.data() + t * m.cap;
                                int64_t i = 0, j = 0, c = 0;
                                while (i < w && j < w) {
                                    if (x[i] == y[j]) { ++c; ++i; ++j; } else if (x[i] < y[j]) ++i; else ++j;
                                }
                                ++queries;
                                same += c == w;
                                if (p0 + t + 1 > m.cap) { ++sel_queries; shared += (double) c; cells += (double) w; }
                            }
                            if (qsa_index + 1 == g.n_qsa_layers())
                                std::fprintf(stderr, "strata prefill: FP16 indexer keys: %lld of %lld selections identical; "
                                                     "where the selection is sparse, %.4f%% of cells shared (%lld queries)\n",
                                             same, queries, cells > 0 ? 100.0 * shared / cells : 100.0, sel_queries);
                        }
                        // STRATA_QSA_DUMP=<file>: append every QSA layer's selected cells for the prompt's last
                        // STRATA_QSA_DUMP_LAST (4096) positions - records of int32 {qsa layer, pos0, T, cap} + T*cap cells,
                        // for tools/qsa_locality.py (how local the sparse attention's reads are: the KV-streaming question)
                        if (static const char* dump = std::getenv("STRATA_QSA_DUMP"); dump != nullptr) {
                            static const long long last = std::getenv("STRATA_QSA_DUMP_LAST")
                                                              ? std::atoll(std::getenv("STRATA_QSA_DUMP_LAST")) : 4096;
                            if (p0 + T > pos0 + n - last) {
                                std::vector<int32_t> h((size_t) (T * m.cap));
                                cudaMemcpyAsync(h.data(), m.sel_ids, h.size() * 4, cudaMemcpyDeviceToHost, m.cs);
                                cudaStreamSynchronize(m.cs);
                                if (std::FILE* f = std::fopen(dump, "ab")) {
                                    const int32_t hdr[4] = {(int32_t) qsa_index, (int32_t) p0, (int32_t) T, (int32_t) m.cap};
                                    std::fwrite(hdr, 4, 4, f);
                                    std::fwrite(h.data(), 4, h.size(), f);
                                    std::fclose(f);
                                }
                            }
                        }
                        const strata::kernels::QsaAttnPools pools = staged ? pools_of(m.stage_pool, m.ident_table)
                                                                           : core::qsa_attn_pools(st);
                        pt.mark(kPfQsaAttn, cs);
                        // the chunk on tensor cores (qsa_prompt_attn.hpp); Q4_0 KV or a pre-sm_80 card: the decode window's
                        // kernel, attn_batch queries at a time
                        if (!strata::kernels::qsa_prompt_attn_batch(m.q, pools, m.sel_ids, m.steps_dev, m.cap, s, m.attn, T, m.cs)) {
                            for (int64_t t0 = 0; t0 < T; t0 += m.attn_batch) {
                                const int64_t nb = std::min(m.attn_batch, T - t0);
                                strata::kernels::qsa_decode_attn_batch(m.q + t0 * ZV, pools, m.sel_ids + t0 * m.cap,
                                                                       m.steps_dev + t0 * strata::kernels::kStepCount, m.cap,
                                                                       s, m.attn_scratch, m.attn + t0 * ZV, nb, m.cs);
                            }
                        }
                        if (st.kv_q4) strata::kernels::fwht256_inplace_cuda(m.attn, T * 24, m.cs);
                        pt.mark(kPfQsa, cs);
                        gate_attn(m.attn, m.Qf, m.attn_h, T, m.cs);
                        if (!native_proj(m.gemm, wo, m.attn_h, m.bo, T, v.name("attn_output.weight"), err)) return false;
                        ++qsa_index;
                    } else {
                        // ======================= MoE =======================
                        const core::WeightRef *wr = need(v, "ffn_gate_inp.weight", err),
                                              *wgi = need(v, "ffn_gate_inp_shexp.weight", err),
                                              *wsg = need(v, "ffn_gate_shexp.weight", err),
                                              *wsu = need(v, "ffn_up_shexp.weight", err),
                                              *wsd = need(v, "ffn_down_shexp.weight", err);
                        if (!wr || !wgi || !wsg || !wsu || !wsd) return false;
                        if (helper) {   // the MoE input to the helper, behind the router and the shared expert
                            cudaEventRecord(m.xa, m.cs);
                            core::DeviceGuard dgh(helper->device);
                            cudaStreamWaitEvent(helper->cs, m.xa, 0);
                            cudaMemcpyPeerAsync(helper->mixed, helper->device, m.mixed, m.device, (size_t) T * N * 4, helper->cs);
                        }
                        pt.mark(kPfRouter, cs);
                        if (!bf16_proj(m.gemm, wr, m.mixed_bf, m.logits, T, v.name("ffn_gate_inp.weight"), err)) return false;
                        route(m.logits, m.ids, m.w, T, m.g->n_expert, m.cs);
                        if (helper) {   // the routing weights to the helper (its partial combine)
                            cudaEventRecord(m.xa, m.cs);
                            core::DeviceGuard dgh(helper->device);
                            cudaStreamWaitEvent(helper->cs, m.xa, 0);
                            cudaMemcpyPeerAsync(helper->w, helper->device, m.w, m.device, (size_t) T * K * 4, helper->cs);
                        }
                        // the shared expert and its scalar gate
                        if (!native_proj(m.gemm, wsg, m.mixed_h, m.sgate, T, v.name("ffn_gate_shexp.weight"), err)) return false;
                        if (!native_proj(m.gemm, wsu, m.mixed_h, m.sup, T, v.name("ffn_up_shexp.weight"), err)) return false;
                        swiglu_pair(m.sgate, m.sup, m.sh_h, T, m.cs);
                        if (!native_proj(m.gemm, wsd, m.sh_h, m.shared, T, v.name("ffn_down_shexp.weight"), err)) return false;
                        if (wgi->kind != core::WeightKind::Bf16InF32) { err = "prefill: shared gate is not BF16"; return false; }
                        m.gemm.bf16(m.mixed_bf, (const uint16_t*) wgi->data, m.sg, T, 1, N);
                        // group the (token, k) pairs by expert on the host: the resident experts' rows first, then the
                        // streamed ones', each part in id order, and an MMQ group never spans the two (the streamed
                        // part's groups are then the same whichever GPU computes them: MMQ's sums depend on a group's
                        // composition)
                        pt.mark(kPfHostGroup, cs);
                        cudaMemcpyAsync(m.ids_host.data(), m.ids, (size_t) T * K * 4, cudaMemcpyDeviceToHost, m.cs);
                        cudaStreamSynchronize(m.cs);
                        pt.fold();
                        std::fill(m.cnt.begin(), m.cnt.end(), 0);
                        for (int64_t i = 0; i < T * K; ++i) {
                            const int32_t e = m.ids_host[(size_t) i];
                            if (e < 0 || e >= m.g->n_expert) { err = "prefill: routed id out of range"; return false; }
                            ++m.cnt[(size_t) e];
                        }
                        auto resident = [&](int32_t e) {
                            return m.resident(l, e);
                        };
                        std::vector<int32_t> order;
                        for (int32_t e = 0; e < m.g->n_expert; ++e) if (m.cnt[(size_t) e] > 0 && resident(e)) order.push_back(e);
                        const size_t n_res = order.size();
                        for (int32_t e = 0; e < m.g->n_expert; ++e) if (m.cnt[(size_t) e] > 0 && !resident(e)) order.push_back(e);
                        const size_t n = order.size();
                        {
                            int32_t row = 0;
                            for (const int32_t e : order) { m.off[(size_t) e] = row; row += m.cnt[(size_t) e]; }
                        }
                        std::vector<int32_t> fill(m.off.begin(), m.off.end() - 1);
                        for (int64_t i = 0; i < T * K; ++i) {
                            const int32_t e = m.ids_host[(size_t) i];
                            const int32_t p = fill[(size_t) e]++;
                            m.slot_host[(size_t) i] = p;
                            m.src_host[(size_t) p] = (int32_t) (i / K);
                        }
                        cudaMemcpyAsync(m.slot_dev, m.slot_host.data(), (size_t) T * K * 4, cudaMemcpyHostToDevice, m.cs);
                        cudaMemcpyAsync(m.src_dev, m.src_host.data(), (size_t) T * K * 4, cudaMemcpyHostToDevice, m.cs);
                        // the MMQ groups: MMQ_GROUP consecutive entries of `order` within its resident or its streamed part
                        std::vector<size_t> gstart;
                        for (size_t j = 0; j < n_res; j += MMQ_GROUP) gstart.push_back(j);
                        const size_t g_res = gstart.size();
                        for (size_t j = n_res; j < n; j += MMQ_GROUP) gstart.push_back(j);
                        const size_t ng = gstart.size();
                        gstart.push_back(n);
                        std::vector<uint32_t> group_of(n);
                        for (size_t gi = 0; gi < ng; ++gi)
                            for (size_t j = gstart[gi]; j < gstart[gi + 1]; ++j) group_of[j] = (uint32_t) gi;
                        const int64_t R0 = n_res < n ? m.off[(size_t) order[n_res]] : T * K;   // the streamed part's first row
                        const bool remote = helper != nullptr && n_res < n;
                        const strata::kernels::cpu::ExpertLayout& lay = strata::kernels::cpu::expert_layout();
                        const bool use_mmq = mmq_plan().any && mmq_plan().layer[(size_t) l];
                        const int mmq_gt = lay.native ? lay.fmt[(size_t) l].gu_type : 42;
                        const int mmq_dt = lay.native ? lay.fmt[(size_t) l].d_type : 42;
                        const size_t mmq_gub = use_mmq ? mmq::matrix_bytes(mmq_gt, 1280, N) : 0;
                        const size_t mmq_db = use_mmq ? mmq::matrix_bytes(mmq_dt, N, 640) : 0;
                        pt.mark(kPfGather, cs);
                        if (use_mmq) {
                            // step 2b: the layer's activations as q8_1 rows in expert order, straight from `mixed` (with
                            // the helper, this GPU's rows only)
                            mmq::quantize(m.mixed, m.src_dev, m.Xq, mmq_gt, N, N, remote ? R0 : T * K, m.cs);
                            // the experts' bounds (absolute rows; gate/up reads the layer's rows), then each group's
                            // relative ones (down reads the group's own quantized H)
                            m.bounds_host.resize(n + 1 + ng * (MMQ_GROUP + 1));
                            for (size_t j = 0; j < n; ++j) m.bounds_host[j] = m.off[(size_t) order[j]];
                            m.bounds_host[n] = (int32_t) (T * K);
                            for (size_t gi = 0; gi < ng; ++gi)
                                for (size_t i = 0; i <= MMQ_GROUP; ++i)
                                    m.bounds_host[n + 1 + gi * (MMQ_GROUP + 1) + i] =
                                        m.bounds_host[std::min(gstart[gi + 1], gstart[gi] + i)] - m.bounds_host[gstart[gi]];
                            cudaMemcpyAsync(m.bounds_dev, m.bounds_host.data(), m.bounds_host.size() * 4,
                                            cudaMemcpyHostToDevice, m.cs);
                            if (remote) {
                                // the helper's rows: their tokens, their q8_1 rows from its copy of `mixed`, the bounds
                                // counted from R0 (the streamed experts', then the streamed groups' relative ones)
                                Impl& h = *helper;
                                h.bounds_host.resize(n - n_res + 1 + (ng - g_res) * (MMQ_GROUP + 1));
                                for (size_t j = n_res; j <= n; ++j) h.bounds_host[j - n_res] = m.bounds_host[j] - (int32_t) R0;
                                std::copy(m.bounds_host.begin() + (std::ptrdiff_t) (n + 1 + g_res * (MMQ_GROUP + 1)), m.bounds_host.end(),
                                          h.bounds_host.begin() + (std::ptrdiff_t) (n - n_res + 1));
                                core::DeviceGuard dgh(h.device);
                                cudaMemcpyAsync(h.src_dev, m.src_host.data() + R0, (size_t) (T * K - R0) * 4, cudaMemcpyHostToDevice, h.cs);
                                cudaMemcpyAsync(h.slot_dev, m.slot_host.data(), (size_t) T * K * 4, cudaMemcpyHostToDevice, h.cs);
                                cudaMemcpyAsync(h.bounds_dev, h.bounds_host.data(), h.bounds_host.size() * 4, cudaMemcpyHostToDevice, h.cs);
                                mmq::quantize(h.mixed, h.src_dev, h.Xq, mmq_gt, N, N, T * K - R0, h.cs);
                            }
                        } else {
                            gather_rows16(m.mixed_h, m.src_dev, m.Xs, T * K, N, m.cs);
                        }
                        // one MMQ group's products on X's buffers: gate/up over its rows of `xq` (quantized as `xq_rows`
                        // rows; expert bounds `gu_b`), swiglu, its H to q8_1, down (group-relative bounds `dn_b`), at the
                        // layer's rows minus `base`
                        auto products = [&](Impl& X, cudaStream_t s, size_t gi, const void* xq, int64_t xq_rows,
                                            const int32_t* gu_b, const int32_t* dn_b, int64_t base) {
                            const size_t j0 = gstart[gi], j1 = gstart[gi + 1];
                            const int ngx = (int) (j1 - j0);
                            const int64_t r0 = m.bounds_host[j0] - base, nr = m.bounds_host[j1] - m.bounds_host[j0];
                            int64_t maxr = 0;
                            for (size_t i = j0; i < j1; ++i) maxr = std::max<int64_t>(maxr, m.cnt[(size_t) order[i]]);
                            // the zeroed tail after the group's last expert (see MMQ_TAIL)
                            cudaMemsetAsync(X.grp_gu + (size_t) ngx * mmq_gub, 0, MMQ_TAIL, s);
                            cudaMemsetAsync(X.grp_d + (size_t) ngx * mmq_db, 0, MMQ_TAIL, s);
                            mmq::Product gu;
                            gu.w = X.grp_gu; gu.type = mmq_gt; gu.w_rows = 1280; gu.w_cols = N; gu.expert_bytes = mmq_gub;
                            gu.n = ngx; gu.xq = xq; gu.bounds = gu_b; gu.ids = X.ids_identity;
                            gu.total_rows = xq_rows; gu.max_rows = maxr; gu.dst = X.GU; gu.ld_dst = 1280;
                            X.mmq_ctx->run(gu, s);
                            mmq::swiglu(X.GU + r0 * 1280, X.H + r0 * 640, nr, 640, !lay.native, s);
                            if (&X == &m) pt.mark(kPfGemmD, cs);
                            mmq::quantize(X.H + r0 * 640, nullptr, X.Hq, mmq_dt, 640, 640, nr, s);
                            mmq::Product dn;
                            dn.w = X.grp_d; dn.type = mmq_dt; dn.w_rows = N; dn.w_cols = 640; dn.expert_bytes = mmq_db;
                            dn.n = ngx; dn.xq = X.Hq; dn.bounds = dn_b; dn.ids = X.ids_identity;
                            dn.total_rows = nr; dn.max_rows = maxr; dn.dst = X.Dm + r0 * N; dn.ld_dst = N;
                            X.mmq_ctx->run(dn, s);
                        };
                        // an expert's blob on X's device into its group slot (GGUF blocks, unchanged or converted)
                        auto gather = [&](Impl& X, cudaStream_t s, size_t j, const uint8_t* blob_dev) {
                            const size_t q = j - gstart[group_of[j]];
                            if (lay.native) {
                                const auto& f = lay.fmt[(size_t) l];
                                mmq::gather_native(blob_dev, blob_dev + f.up_off, mmq_gub / 2, blob_dev + f.down_off, mmq_db,
                                                   X.grp_gu + q * mmq_gub, X.grp_d + q * mmq_db, s);
                            } else {
                                mmq::gather_strata_q2(blob_dev, X.grp_gu + q * mmq_gub, X.grp_d + q * mmq_db, s);
                            }
                        };
                        // Stage ahead: the copy stream moves blobs host -> device while the compute stream works.
                        int stage_next = 0;
                        std::vector<int> stage_of(order.size(), -1);
                        // the unpinned ones are copied to pinned buffers by the stager's threads, in this order
                        std::vector<int> job_of(order.size(), -1);
                        if (!stream_all && !helper) {
                            std::vector<Stager::Job> js;
                            for (size_t j = 0; j < order.size(); ++j) {
                                const int32_t e = order[j];
                                if (m.resident(l, e)) continue;
                                if (m.src->pinned(l, e)) continue;
                                const uint8_t* b = m.src->blob(l, e);
                                if (!b) { err = "prefill: expert source has no blob"; return false; }
                                job_of[j] = (int) js.size();
                                js.push_back({b, (size_t) lay.blob_bytes(l)});
                            }
                            m.stager->start(std::move(js));
                        }
                        StagerDone stager_done{stream_all || helper ? nullptr : m.stager.get()};
                        auto stage_one = [&](size_t j) -> bool {
                            const int32_t e = order[j];
                            const bool resident = m.resident(l, e);
                            if (resident) return true;
                            const int sl = stage_next;
                            stage_next = (stage_next + 1) % STAGE;
                            const auto th = Clock::now();
                            const uint8_t* b = m.src->blob(l, e);
                            if (!b) { err = "prefill: expert source has no blob"; return false; }
                            if (m.src->pinned(l, e)) {
                                // DMA straight from the page-locked arena: the copy stream only waits for the slot
                                if (m.stage_live[sl]) cudaStreamWaitEvent(m.copy, m.used[sl], 0);
                                cudaMemcpyAsync(m.stage_dev[sl], b, (size_t) lay.blob_bytes(l), cudaMemcpyHostToDevice, m.copy);
                                ++pst.experts_dma;
                            } else {
                                // copied to a pinned buffer by the stager (waits only if it is behind), then DMA
                                const uint8_t* hb = m.stager->wait(job_of[j]);
                                if (m.stage_live[sl]) cudaStreamWaitEvent(m.copy, m.used[sl], 0);
                                cudaMemcpyAsync(m.stage_dev[sl], hb, (size_t) lay.blob_bytes(l), cudaMemcpyHostToDevice, m.copy);
                                m.stager->issued_one(job_of[j], m.copy);
                            }
                            cudaEventRecord(m.copied[sl], m.copy);
                            m.stage_live[sl] = true;
                            stage_of[j] = sl;
                            pst.ms_experts_host += ms_since(th);
                            ++pst.experts_streamed;
                            return true;
                        };
                        // one expert's products from its blob on the device; `slot` (a ring slot, or -1 for a resident
                        // expert) is released once the blob is read
                        auto compute = [&](size_t j, const uint8_t* blob_dev, int slot) -> bool {
                            const int32_t e = order[j];
                            pt.mark(kPfDequant, cs);
                            if (use_mmq) {
                                gather(m, m.cs, j, blob_dev);
                                if (slot >= 0) cudaEventRecord(m.used[slot], m.cs);
                                const size_t gi = group_of[j];
                                if (j + 1 < gstart[gi + 1]) return true;
                                pt.mark(kPfGemmGU, cs);
                                products(m, m.cs, gi, m.Xq, remote ? R0 : T * K, m.bounds_dev + gstart[gi],
                                         m.bounds_dev + n + 1 + gi * (MMQ_GROUP + 1), 0);
                                return true;
                            }
                            const int q = (int) (j % DQ);
                            if (lay.native) {
                                // plan v0.3 P6: a native pack's layer, dequantized by llama.cpp's own formulas
                                const auto& f = lay.fmt[(size_t) l];
                                strata::kernels::iq_dequant_gu_f16(f.gu_type, blob_dev, blob_dev + f.up_off, f.n_ff, f.n_embd,
                                                                   m.dq_gu[q], m.cs);
                                strata::kernels::iq_dequant_f16(f.d_type, blob_dev + f.down_off, f.n_embd * f.n_ff, m.dq_d[q], m.cs);
                            } else {
                                blob_dequant_f16(blob_dev, m.dq_gu[q], m.dq_d[q], m.cs);
                            }
                            if (slot >= 0) cudaEventRecord(m.used[slot], m.cs);
                            const int64_t o0 = m.off[(size_t) e], ne = m.cnt[(size_t) e];
                            pt.mark(kPfGemmGU, cs);
                            m.gemm.f16(m.Xs + o0 * N, m.dq_gu[q], m.GU + o0 * 1280, ne, 1280, N);
                            swiglu_interleaved(m.GU + o0 * 1280, m.Hh + o0 * 640, ne, m.cs);
                            pt.mark(kPfGemmD, cs);
                            m.gemm.f16(m.Hh + o0 * 640, m.dq_d[q], m.Dm + o0 * N, ne, N, 640);
                            return true;
                        };
                        if (!stream_all && !helper) {
                            size_t staged = 0;
                            const size_t lookahead = STAGE - 1;
                            for (size_t j = 0; j < order.size(); ++j) {
                                while (staged < order.size() && staged <= j + lookahead) {
                                    if (!stage_one(staged)) return false;
                                    ++staged;
                                }
                                const int32_t e = order[j];
                                if (stage_of[j] < 0) {
                                    ++pst.experts_resident;
                                    if (!compute(j, (const uint8_t*) m.slot_addr[m.host_res[(size_t) l * m.g->n_expert + e]], -1)) return false;
                                } else {
                                    pt.mark(kPfWaitCopy, cs);
                                    cudaStreamWaitEvent(m.cs, m.copied[stage_of[j]], 0);
                                    if (!compute(j, m.stage_dev[stage_of[j]], stage_of[j])) return false;
                                }
                            }
                        } else if (helper) {
                            // the streamed part on the helper, queued first: this layer's entries of its stream in id
                            // order (an entry the routing did not pick only gives its slot back); then its partial
                            // combine comes back on this GPU's copy stream, into `emb` (idle once the chunk's embedding
                            // is broadcast)
                            Impl& h = *helper;
                            size_t k = seq_start[(size_t) l];
                            const size_t kend = seq_start[(size_t) l + 1];
                            auto release_to = [&](int32_t e_stop) {
                                while (k < kend && hseq[k].e < e_stop) {
                                    {
                                        core::DeviceGuard dgh(h.device);
                                        cudaEventRecord(h.used[k % (size_t) h.ring], h.cs);
                                    }
                                    h_consumed = ++k;
                                    h_issue_until(h_consumed + (size_t) h.ring);
                                }
                            };
                            for (size_t j = n_res; j < n; ++j) {
                                const int32_t e = order[j];
                                release_to(e);
                                const int sl = (int) (k % (size_t) h.ring);
                                const size_t gi = group_of[j];
                                {
                                    core::DeviceGuard dgh(h.device);
                                    cudaStreamWaitEvent(h.cs, h.copied[sl], 0);
                                    gather(h, h.cs, j, h.stage_dev[sl]);
                                    cudaEventRecord(h.used[sl], h.cs);
                                    if (j + 1 == gstart[gi + 1])
                                        products(h, h.cs, gi, h.Xq, T * K - R0, h.bounds_dev + (gstart[gi] - n_res),
                                                 h.bounds_dev + (n - n_res + 1) + (gi - g_res) * (MMQ_GROUP + 1), R0);
                                }
                                h_consumed = ++k;
                                h_issue_until(h_consumed + (size_t) h.ring);
                            }
                            release_to(m.g->n_expert);
                            if (remote) {
                                {
                                    core::DeviceGuard dgh(h.device);
                                    moe_partial(h.Dm, h.slot_dev, h.w, h.bo, T, R0, h.cs);
                                    cudaEventRecord(h.xa, h.cs);
                                }
                                cudaStreamWaitEvent(m.copy, h.xa, 0);
                                cudaMemcpyPeerAsync(m.emb, m.device, h.bo, h.device, (size_t) T * N * 4, m.copy);
                                cudaEventRecord(m.xb, m.copy);
                            }
                            // the resident part from VRAM, while the helper works
                            for (size_t j = 0; j < n_res; ++j) {
                                ++pst.experts_resident;
                                if (!compute(j, (const uint8_t*) m.slot_addr[m.host_res[(size_t) l * m.g->n_expert + order[j]]], -1)) return false;
                            }
                            if (remote) cudaStreamWaitEvent(m.cs, m.xb, 0);
                        } else {
                            // the streamed walk first (its slots go back to the ring for the next layers' copies): this
                            // layer's entries [k, kend) in id order, an entry the routing did not pick only gives its
                            // slot back; then the resident part from VRAM
                            size_t k = seq_start[(size_t) l];
                            const size_t kend = seq_start[(size_t) l + 1];
                            auto release_to = [&](int32_t e_stop) {
                                while (k < kend && seq[k].e < e_stop) {
                                    cudaEventRecord(m.used[k % (size_t) m.ring], m.cs);
                                    consumed = ++k;
                                    issue_until(consumed + (size_t) m.ring);
                                }
                            };
                            for (size_t j = n_res; j < n; ++j) {
                                const int32_t e = order[j];
                                release_to(e);
                                const int sl = (int) (k % (size_t) m.ring);
                                pt.mark(kPfWaitCopy, cs);
                                cudaStreamWaitEvent(m.cs, m.copied[sl], 0);
                                if (!compute(j, m.stage_dev[sl], sl)) return false;
                                consumed = ++k;
                                issue_until(consumed + (size_t) m.ring);
                            }
                            release_to(m.g->n_expert);
                            for (size_t j = 0; j < n_res; ++j) {
                                ++pst.experts_resident;
                                if (!compute(j, (const uint8_t*) m.slot_addr[m.host_res[(size_t) l * m.g->n_expert + order[j]]], -1)) return false;
                            }
                        }
                        pt.mark(kPfCombine, cs);
                        moe_combine(m.Dm, m.slot_dev, m.w, m.shared, m.sg, m.bo, T, R0, remote ? m.emb : nullptr, m.cs);
                        // debug: STRATA_DBG_NAN=1 reports the first layer of a chunk whose MoE produced non-finite values
                        if (static const bool dbg = std::getenv("STRATA_DBG_NAN") != nullptr; dbg) {
                            cudaStreamSynchronize(m.cs);
                            auto bad = [&](const float* d, int64_t n) {
                                std::vector<float> h((size_t) n);
                                cudaMemcpy(h.data(), d, (size_t) n * 4, cudaMemcpyDeviceToHost);
                                int64_t c = 0;
                                for (float v : h) c += !std::isfinite(v);
                                return c;
                            };
                            const int64_t bgu = bad(m.GU, T * K * 1280), bdm = bad(m.Dm, T * K * N), bbo = bad(m.bo, T * N);
                            const int64_t bh = m.H ? bad(m.H, T * K * 640) : -1;
                            static int64_t reported = -1;
                            if ((bgu || bdm || bbo || bh > 0) && reported != pst.chunks) {
                                reported = pst.chunks;
                                std::fprintf(stderr, "strata dbg: layer %lld (mmq %d, types %d/%d, %zu experts): non-finite GU %lld "
                                             "H %lld Dm %lld bo %lld of T %lld\n", (long long) l, (int) use_mmq, mmq_gt, mmq_dt,
                                             order.size(), (long long) bgu, (long long) bh, (long long) bdm, (long long) bbo,
                                             (long long) T);
                            }
                        }
                    }
                    // ---- the hyper-connection write of this half
                    gr_write(m.R, m.bo, m.inj, HC, T, m.cs);
                    if (half == 1 && strata::kernels::cvec().covers(l))   // --control-vector-scaled
                        strata::kernels::cvec_apply(m.R, l, T, D, nullptr, 0, nullptr, 0, false, m.cs);
                }
            }
            if (pi + 1 < n_parts) {
                cudaEventRecord(m.done, cs);
                signal(done, pi, k + 1);
                if (pi == 0) hold = hold_at && hold_at(p0 + T);
                continue;
            }
            // ---- the last stage: the chunk's final residuals
            pst.tokens += T;
            pt.mark(kPfStart, cs);
            if (const char* dump = std::getenv("STRATA_PREFILL_DUMP_R")) {   // debug: the final residuals, every 64th
                cudaStreamSynchronize(m.cs);                                  // position (A/B quality of this path)
                if (std::FILE* f = std::fopen(dump, c0 == 0 ? "wb" : "ab")) {
                    std::vector<float> row((size_t) D);
                    for (int64_t t = (64 - p0 % 64) % 64; t < T; t += 64) {
                        cudaMemcpy(row.data(), m.R + t * D, (size_t) D * 4, cudaMemcpyDeviceToHost);
                        const int64_t pos = p0 + t;
                        std::fwrite(&pos, sizeof pos, 1, f);
                        std::fwrite(row.data(), 4, row.size(), f);
                    }
                    std::fclose(f);
                }
            }
            if (on_chunk) {
                if (cudaStreamSynchronize(m.cs) != cudaSuccess) {
                    err = std::string("prefill: ") + cudaGetErrorString(cudaGetLastError());
                    return false;
                }
                if (!on_chunk(m.R, T, p0, m.region, (size_t) m.region_bytes, err)) return false;
            }
            signal(done, pi, k + 1);
            if (pi == 0) hold = false;
        }
        return true;
    };
    // the first stage on this thread, the others on their own
    std::vector<std::thread> threads;
    for (size_t pi = 1; pi < n_parts; ++pi)
        threads.emplace_back([&, pi] {
            std::string e;
            if (!run_part(pi, e)) fail(e);
        });
    {
        std::string e;
        if (!run_part(0, e)) fail(e);
    }
    for (std::thread& t : threads) t.join();
    for (const PrefillStats& p : part_stats) {
        stats_.tokens += p.tokens;
        stats_.chunks = std::max(stats_.chunks, p.chunks);
        stats_.ms_experts_host += p.ms_experts_host;
        stats_.experts_streamed += p.experts_streamed;
        stats_.experts_dma += p.experts_dma;
        stats_.experts_resident += p.experts_resident;
        stats_.experts_helper += p.experts_helper;
        stats_.ms_ple += p.ms_ple;
    }
    if (aborted) {
        for (auto& part : parts_) {
            core::DeviceGuard dg(part->device);
            cudaStreamSynchronize(part->cs);
        }
        err = first_err;
        return false;
    }
    ss.ple_prev[0] = prev[0];
    ss.ple_prev[1] = prev[1];
    for (auto& part : parts_)
        if (cudaStreamSynchronize(part->cs) != cudaSuccess) {
            err = std::string("prefill: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
    if (std::getenv("STRATA_DBG_NAN") != nullptr) {   // debug: the state the prompt leaves for the token path
        auto bad = [&](const float* d, int64_t n) {
            std::vector<float> h((size_t) n);
            cudaMemcpy(h.data(), d, (size_t) n * 4, cudaMemcpyDeviceToHost);
            int64_t c = 0;
            double mx = 0;
            for (float v : h) { c += !std::isfinite(v); if (std::isfinite(v)) mx = std::max(mx, (double) std::fabs(v)); }
            std::fprintf(stderr, " %lld non-finite (max |x| %.3g)", (long long) c, mx);
        };
        const int64_t last = (n - 1) % m0.T;
        std::fprintf(stderr, "strata dbg: prompt end: last residual row");
        bad(ml.R + last * D, D);
        if (ss.ple.ready()) { std::fprintf(stderr, "; PLE history"); bad(ss.ple.hist, (int64_t) strata::kernels::NG_HIST * strata::kernels::NG_HC_DIM); }
        std::fprintf(stderr, "; GDN state 0");
        bad(ss.gdn_states[0], 64 * 1024);
        std::fprintf(stderr, "\n");
    }
    stats_.ms_total += ms_since(t_start);
    for (PfTimer& pt : timers) {
        if (!pt.on) break;
        pt.fold();
        double total = 0.0;
        for (double v : pt.ms) total += v;
        std::string line;
        char b[96];
        for (int i = 0; i < kPfCount; ++i) {
            if (pt.ms[i] <= 0.0) continue;
            std::snprintf(b, sizeof b, " %s %.0f (%.1f%%)", kPfNames[i], pt.ms[i], total > 0 ? 100.0 * pt.ms[i] / total : 0.0);
            line += b;
        }
        std::fprintf(stderr, "strata prefill timing: %lld tokens, GPU timeline %.0f ms, wall %.0f ms, host staging %.0f ms:%s\n",
                     (long long) n, total, ms_since(t_start), stats_.ms_experts_host, line.c_str());
    }
    if (std::getenv("STRATA_STATE_HASH_GDN") != nullptr) {   // debug: the GDN states as the prompt path leaves them
        std::vector<uint8_t> b((size_t) gdn_floats * 4);
        std::string line;
        char h[8];
        for (int64_t i = 0; i < g.n_gdn_layers(); ++i) {
            cudaMemcpy(b.data(), ss.gdn_states[(size_t) i], b.size(), cudaMemcpyDeviceToHost);
            uint64_t x = 1469598103934665603ull;
            for (uint8_t c : b) x = (x ^ c) * 1099511628211ull;
            std::snprintf(h, sizeof(h), "%04llx ", (unsigned long long) (x & 0xffff));
            line += h;
        }
        std::fprintf(stderr, "strata prefill: GDN_HASH %s\n", line.c_str());
    }
    return true;
}

}  // namespace strata::prefill
