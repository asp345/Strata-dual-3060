// src/kernels/cuda/fused_gr.cu - see include/strata/kernels/fused_gr.hpp.
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/bf16_bits.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int N = 2560;         // n_embd
constexpr int HC = 4;           // streams
constexpr int D = N * HC;       // 10240
constexpr int LR = 320;         // hc_lr
constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;
constexpr int DOWN_BLOCKS = LR / WARPS;          // 40 blocks of 8 rows; one more for the inject rows
constexpr int UP_COLS = 32;                      // columns d per `up` block (x 4 streams = 128 rows)
constexpr int UP_BLOCKS = N / UP_COLS;           // 80

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + __expf(-x)); }

// 8 bf16 packed in a uint4 against 8 floats.
__device__ __forceinline__ float dot8(const uint4 w, const float* x) {
    float acc = 0.0f;
    const uint32_t v[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        acc = fmaf(__uint_as_float(v[j] << 16), x[2 * j], acc);
        acc = fmaf(__uint_as_float(v[j] & 0xffff0000u), x[2 * j + 1], acc);
    }
    return acc;
}

__global__ void __launch_bounds__(THREADS) gr_down_kernel(FusedGrArgs a) {
    __shared__ __align__(16) float xn[D];
    __shared__ float part[WARPS][HC];
    __shared__ float s_rs[HC];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    float gw[HC];
#pragma unroll
    for (int c = 0; c < HC; ++c) gw[c] = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC) : 0.0f;
    // 1. R' * w_norm into shared memory, and the per-stream sums of squares of R'.
    float ss[HC] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = t * 4; i < D; i += THREADS * 4) {
        const int c = i / N, d = i - c * N;
        float4 r = *reinterpret_cast<const float4*>(a.R + i);
        if (a.apply) {
            const float4 b = *reinterpret_cast<const float4*>(a.bo_prev + d);
            r.x = fmaf(b.x, gw[c], r.x); r.y = fmaf(b.y, gw[c], r.y);
            r.z = fmaf(b.z, gw[c], r.z); r.w = fmaf(b.w, gw[c], r.w);
        }
        const float4 g = *reinterpret_cast<const float4*>(a.w_norm + i);
        float sq = r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w;
#pragma unroll
        for (int cc = 0; cc < HC; ++cc) if (cc == c) ss[cc] += sq;
        *reinterpret_cast<float4*>(xn + i) = make_float4(r.x * g.x, r.y * g.y, r.z * g.z, r.w * g.w);
    }
#pragma unroll
    for (int c = 0; c < HC; ++c) {
        const float v = warp_sum(ss[c]);
        if (lane == 0) part[warp][c] = v;
    }
    __syncthreads();
    if (t < HC) {
        float s = 0.0f;
        for (int w = 0; w < WARPS; ++w) s += part[w][t];
        s_rs[t] = rsqrtf(s / (float) N + a.eps);
        if (blockIdx.x == 0) a.rs[t] = s_rs[t];
    }
    __syncthreads();
    for (int i = t; i < D; i += THREADS) xn[i] *= s_rs[i / N];
    __syncthreads();
    // 2. one warp per output row: 10240 bf16 = 1280 chunks of 8, 40 per lane.
    const bool inject_block = blockIdx.x == DOWN_BLOCKS;
    const int row = inject_block ? warp : blockIdx.x * WARPS + warp;
    if (inject_block && (a.w_inject == nullptr || warp >= HC)) return;
    const uint16_t* wrow = (inject_block ? a.w_inject : a.w_down) + (size_t) row * D;
    const uint4* w4 = reinterpret_cast<const uint4*>(wrow);
    float acc = 0.0f;
#pragma unroll 4
    for (int j = lane; j < D / 8; j += 32) acc += dot8(__ldg(w4 + j), xn + j * 8);
    acc = warp_sum(acc);
    if (lane != 0) return;
    if (inject_block) {
        a.inject_out[row] = acc;
    } else {
        const float x = acc / (float) HC;
        a.lo[row] = x / (1.0f + __expf(-x));
    }
}

__global__ void __launch_bounds__(THREADS) gr_up_kernel(FusedGrArgs a) {
    __shared__ __align__(16) float lo[LR];
    __shared__ float g[HC][UP_COLS];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int d0 = blockIdx.x * UP_COLS;
    for (int k = t; k < LR; k += THREADS) lo[k] = a.lo[k];
    __syncthreads();
    // 128 rows (4 streams x 32 columns), 16 per warp: 320 bf16 = 40 chunks of 8.
    for (int r = warp; r < HC * UP_COLS; r += WARPS) {
        const int c = r / UP_COLS, dd = r - c * UP_COLS, i = c * N + d0 + dd;
        const uint4* w4 = reinterpret_cast<const uint4*>(a.w_up + (size_t) i * LR);
        float acc = dot8(__ldg(w4 + lane), lo + lane * 8);
        if (lane < LR / 8 - 32) acc += dot8(__ldg(w4 + 32 + lane), lo + (32 + lane) * 8);
        acc = warp_sum(acc);
        if (lane == 0) {
            float rv = a.R[i];
            if (a.apply) {
                rv = fmaf(a.bo_prev[d0 + dd], 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC), rv);
                a.R_out[i] = rv;                       // this block owns column d0+dd of every stream
            }
            const float x = rv * a.w_norm[i] * a.rs[c];
            g[c][dd] = x * sigmoidf_(acc);
        }
    }
    __syncthreads();
    if (t < UP_COLS) {
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) s += g[c][t];
        a.mixed[d0 + t] = s / (float) HC;
    }
}

// ================================ plan v0.3 P6: T tokens, one weight read ================================
struct GrMulti {
    FusedGrArgs a[kFusedGrMaxT];
    float* xn;
    float* part;   // [KQ][T][LR + HC] partial dots of gr_down
    int T;
};

// warp_sum of v[0 .. P) (P = 1, 2, 4 or 8 tokens) with the shuffles shared: in each of the first log2(P) stages a lane
// keeps half of its tokens and trades the other half with its partner, then the single value finishes the stages.
// Every add is the one warp_sum makes (own + partner, and a + b == b + a), so each token's sum is bit for bit
// warp_sum's.  Each lane ends with one token's sum; token_lane(k) is a lane that holds token k's.
template <int P>
__device__ __forceinline__ float warp_sum_split(const float (&v)[P], int lane) {
    constexpr int L = P == 1 ? 0 : P == 2 ? 1 : P == 4 ? 2 : P == 8 ? 3 : P == 16 ? 4 : 5;
    float cur[P];
#pragma unroll
    for (int i = 0; i < P; ++i) cur[i] = v[i];
#pragma unroll
    for (int st = 0; st < L; ++st) {
        const int o = 16 >> st;
        const bool upper = (lane & o) != 0;
        const int half = (P >> st) / 2;
#pragma unroll
        for (int i = 0; i < half; ++i) {
            const float send = upper ? cur[i] : cur[half + i];
            const float keep = upper ? cur[half + i] : cur[i];
            cur[i] = keep + __shfl_xor_sync(0xffffffffu, send, o);
        }
    }
    float x = cur[0];
#pragma unroll
    for (int o = 16 >> L; o > 0; o >>= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
    return x;
}
// the lane that holds token k's sum after warp_sum_split<P>
template <int P>
__device__ __forceinline__ int token_lane(int k) {
    constexpr int L = P == 1 ? 0 : P == 2 ? 1 : P == 4 ? 2 : P == 8 ? 3 : P == 16 ? 4 : 5;
    int src = 0;
#pragma unroll
    for (int st = 0; st < L; ++st) src |= ((k >> (L - 1 - st)) & 1) << (4 - st);
    return src;
}
// token `lane`'s warp sum in lane `lane` (lanes >= T get an unused value)
template <int P>
__device__ __forceinline__ float warp_sums_to_lanes(const float (&v)[P], int lane) {
    const float x = warp_sum_split<P>(v, lane);
    return __shfl_sync(0xffffffffu, x, token_lane<P>(lane & (P - 1)));
}

__device__ __forceinline__ float dot8h(const uint4 w, const float4 a, const float4 b) {
    float acc = 0.0f;
    acc = fmaf(__uint_as_float(w.x << 16), a.x, acc);
    acc = fmaf(__uint_as_float(w.x & 0xffff0000u), a.y, acc);
    acc = fmaf(__uint_as_float(w.y << 16), a.z, acc);
    acc = fmaf(__uint_as_float(w.y & 0xffff0000u), a.w, acc);
    acc = fmaf(__uint_as_float(w.z << 16), b.x, acc);
    acc = fmaf(__uint_as_float(w.z & 0xffff0000u), b.y, acc);
    acc = fmaf(__uint_as_float(w.w << 16), b.z, acc);
    acc = fmaf(__uint_as_float(w.w & 0xffff0000u), b.w, acc);
    return acc;
}

// Step 1 of `gr_down_kernel`: rs and xn to global.  Block (token k, stream c): the stream's squares with the thread-to-element mapping of a whole-token block (thread t
// takes elements 4t + 4 * THREADS * j, those of stream c), so each partial sum, the warp sums and the sum over warps
// are the ones that block computes, bit for bit - but the token's four streams now run in four blocks.
__global__ void __launch_bounds__(THREADS) gr_norm_multi_kernel(GrMulti m) {
    __shared__ float part[WARPS];
    __shared__ float s_rs;
    const int k = blockIdx.x, c = blockIdx.y;
    const FusedGrArgs& a = m.a[k];
    float* xn = m.xn + (size_t) k * D;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const float gw = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC) : 0.0f;
    float ss = 0.0f;
    const int j0 = (c * N) / (THREADS * 4), j1 = ((c + 1) * N + THREADS * 4 - 1) / (THREADS * 4);
    for (int j = j0; j < j1; ++j) {
        const int i = t * 4 + j * THREADS * 4;
        if (i < c * N || i >= (c + 1) * N) continue;
        const int d = i - c * N;
        float4 r = *reinterpret_cast<const float4*>(a.R + i);
        if (a.apply) {
            const float4 b = *reinterpret_cast<const float4*>(a.bo_prev + d);
            r.x = fmaf(b.x, gw, r.x); r.y = fmaf(b.y, gw, r.y);
            r.z = fmaf(b.z, gw, r.z); r.w = fmaf(b.w, gw, r.w);
        }
        const float4 g = *reinterpret_cast<const float4*>(a.w_norm + i);
        ss += r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w;
        *reinterpret_cast<float4*>(xn + i) = make_float4(r.x * g.x, r.y * g.y, r.z * g.z, r.w * g.w);
    }
    const float v = warp_sum(ss);
    if (lane == 0) part[warp] = v;
    __syncthreads();
    if (t == 0) {
        float s = 0.0f;
        for (int w = 0; w < WARPS; ++w) s += part[w];
        s_rs = rsqrtf(s / (float) N + a.eps);
        a.rs[c] = s_rs;
    }
    __syncthreads();
    for (int i = c * N + t; i < (c + 1) * N; i += THREADS) xn[i] *= s_rs;
}

constexpr int KQ = 4;                  // gr_down splits each row's D inputs into KQ quarters, one block column each
constexpr int KC = D / KQ;             // 2560
constexpr int KCQ = KC / 8 / 32;       // uint4 weight chunks per lane: 10
constexpr int ROWS_D = LR + HC;        // the down rows, then the inject rows

// Step 2 for T tokens: warp (row, quarter) loads its 10 weight chunks at once and reads xn through L1; lane k writes
// token k's partial dot.  The quarters are summed, in order, by gr_up_multi_kernel.
__global__ void __launch_bounds__(THREADS) gr_down_multi_kernel(GrMulti m) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, T = m.T;
    const int row = blockIdx.x * WARPS + warp, q = blockIdx.y;
    if (row >= ROWS_D || (row >= LR && m.a[0].w_inject == nullptr)) return;
    const uint16_t* wrow = row < LR ? m.a[0].w_down + (size_t) row * D : m.a[0].w_inject + (size_t) (row - LR) * D;
    const uint4* w4 = reinterpret_cast<const uint4*>(wrow + q * KC);
    uint4 wv[KCQ];
#pragma unroll
    for (int j = 0; j < KCQ; ++j) wv[j] = __ldg(w4 + lane + 32 * j);
    float acc[kFusedGrMaxT];
#pragma unroll
    for (int k = 0; k < kFusedGrMaxT; ++k) acc[k] = 0.0f;
    const float4* x4 = reinterpret_cast<const float4*>(m.xn + q * KC);
#pragma unroll
    for (int j = 0; j < KCQ; ++j) {
        const int c = lane + 32 * j;
#pragma unroll
        for (int k = 0; k < kFusedGrMaxT; ++k)
            if (k < T) acc[k] += dot8h(wv[j], __ldg(x4 + k * (D / 4) + 2 * c), __ldg(x4 + k * (D / 4) + 2 * c + 1));
    }
    float mine;
    if (T == 1) {
        float v1[1] = {acc[0]};
        mine = warp_sums_to_lanes<1>(v1, lane);
    } else if (T == 2) {
        float v2[2] = {acc[0], acc[1]};
        mine = warp_sums_to_lanes<2>(v2, lane);
    } else if (T <= 4) {
        float v4[4] = {acc[0], acc[1], acc[2], T > 3 ? acc[3] : 0.0f};
        mine = warp_sums_to_lanes<4>(v4, lane);
    } else {
        float v8[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) v8[k] = k < T ? acc[k] : 0.0f;
        mine = warp_sums_to_lanes<8>(v8, lane);
    }
    if (lane < T) m.part[((size_t) q * T + lane) * ROWS_D + row] = mine;
}

constexpr int UPM_COLS = 16;                      // columns per block (x 4 streams = 64 rows, 8 per warp)
constexpr int UPM_BLOCKS = N / UPM_COLS;          // 160

// `gr_up_kernel` for T tokens.  The prologue finishes gr_down: lo = silu((sum of the quarters, in order) / HC), and
// block 0 writes lo and inject_out.  Each row of w_up is read once; the T dots are reduced by xor so every lane holds
// every sum, and lane k runs token k's epilogue.
__global__ void __launch_bounds__(THREADS) gr_up_multi_kernel(GrMulti m) {
    // lo as two float4-half arrays per token: chunk c (8 floats) at loh[k][0][c] and loh[k][1][c], so a lane's reads
    // are 16 bytes apart (no bank conflicts).  The epilogue's inputs for the block's 64 rows are staged once,
    // coalesced.
    __shared__ __align__(16) float4 loh[kFusedGrMaxT][2][LR / 8];
    __shared__ float g[kFusedGrMaxT][HC][UPM_COLS];
    __shared__ float rv_s[kFusedGrMaxT][HC][UPM_COLS], wn_s[HC][UPM_COLS], bo_s[kFusedGrMaxT][UPM_COLS];
    __shared__ float rs_s[kFusedGrMaxT][HC], ip_s[kFusedGrMaxT][HC];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int T = m.T;
    const int d0 = blockIdx.x * UPM_COLS;
    for (int i = t; i < T * ROWS_D; i += THREADS) {
        const int k = i / ROWS_D, r = i - k * ROWS_D;
        float s = m.part[(size_t) k * ROWS_D + r];
#pragma unroll
        for (int q = 1; q < KQ; ++q) s += m.part[((size_t) q * T + k) * ROWS_D + r];
        if (r < LR) {
            const float x = s / (float) HC;
            const float lo = x / (1.0f + __expf(-x));
            reinterpret_cast<float*>(&loh[k][(r >> 2) & 1][r >> 3])[r & 3] = lo;
            if (blockIdx.x == 0) m.a[k].lo[r] = lo;
        } else if (blockIdx.x == 0 && m.a[0].w_inject != nullptr) {
            m.a[k].inject_out[r - LR] = s;
        }
    }
    for (int i = t; i < T * HC * UPM_COLS; i += THREADS) {
        const int k = i / (HC * UPM_COLS), r = i - k * (HC * UPM_COLS), c = r / UPM_COLS, dd = r - c * UPM_COLS;
        rv_s[k][c][dd] = m.a[k].R[c * N + d0 + dd];
    }
    for (int i = t; i < HC * UPM_COLS; i += THREADS) wn_s[i / UPM_COLS][i % UPM_COLS] = m.a[0].w_norm[(i / UPM_COLS) * N + d0 + i % UPM_COLS];
    for (int i = t; i < T * UPM_COLS; i += THREADS) {
        const int k = i / UPM_COLS;
        if (m.a[k].apply) bo_s[k][i % UPM_COLS] = m.a[k].bo_prev[d0 + i % UPM_COLS];
    }
    for (int i = t; i < T * HC; i += THREADS) {
        const int k = i / HC;
        rs_s[k][i % HC] = m.a[k].rs[i % HC];
        if (m.a[k].apply) ip_s[k][i % HC] = m.a[k].inj_prev[i % HC];
    }
    __syncthreads();
    auto wrow = [&](int r) {
        const int c = r / UPM_COLS, dd = r - c * UPM_COLS;
        return reinterpret_cast<const uint4*>(m.a[0].w_up + (size_t) (c * N + d0 + dd) * LR);
    };
    for (int r = warp; r < HC * UPM_COLS; r += WARPS) {
        const int c = r / UPM_COLS, dd = r - c * UPM_COLS, i = c * N + d0 + dd;
        const uint4 wa = __ldg(wrow(r) + lane);
        const uint4 wb = lane < LR / 8 - 32 ? __ldg(wrow(r) + 32 + lane) : make_uint4(0, 0, 0, 0);
        float acc[kFusedGrMaxT];
#pragma unroll
        for (int k = 0; k < kFusedGrMaxT; ++k) {
            acc[k] = 0.0f;
            if (k < T) {
                acc[k] = dot8h(wa, loh[k][0][lane], loh[k][1][lane]);
                if (lane < LR / 8 - 32) acc[k] += dot8h(wb, loh[k][0][32 + lane], loh[k][1][32 + lane]);
            }
        }
        float mine;
        if (T == 1) {
            float v1[1] = {acc[0]};
            mine = warp_sums_to_lanes<1>(v1, lane);
        } else if (T == 2) {
            float v2[2] = {acc[0], acc[1]};
            mine = warp_sums_to_lanes<2>(v2, lane);
        } else if (T <= 4) {
            float v4[4] = {acc[0], acc[1], acc[2], acc[3]};
            mine = warp_sums_to_lanes<4>(v4, lane);
        } else {
            float v8[8];
#pragma unroll
            for (int k = 0; k < 8; ++k) v8[k] = acc[k];
            mine = warp_sums_to_lanes<8>(v8, lane);
        }
        if (lane < T) {
            float rv = rv_s[lane][c][dd];
            if (m.a[lane].apply) {
                rv = fmaf(bo_s[lane][dd], 2.0f * sigmoidf_(ip_s[lane][c] / (float) HC), rv);
                m.a[lane].R_out[i] = rv;
            }
            const float x = rv * wn_s[c][dd] * rs_s[lane][c];
            g[lane][c][dd] = x * sigmoidf_(mine);
        }
    }
    __syncthreads();
    for (int i = t; i < T * UPM_COLS; i += THREADS) {
        const int k = i / UPM_COLS, col = i - k * UPM_COLS;
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) s += g[k][c][col];
        m.a[k].mixed[d0 + col] = s / (float) HC;
    }
}

__global__ void gr_write_fused_kernel(float* __restrict__ R, const float* __restrict__ bo,
                                     const float* __restrict__ inj, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t t = i / D;
    const int j = (int) (i - t * D), c = j / N, d = j - c * N;
    R[i] = fmaf(bo[t * N + d], 2.0f * sigmoidf_(inj[t * HC + c] / (float) HC), R[i]);
}

}  // namespace

void fused_gr_read_multi(const FusedGrArgs* a, int n_tok, float* xn_scratch, void* stream) {
    if (n_tok < 1 || n_tok > kFusedGrMaxT || xn_scratch == nullptr) {
        std::fprintf(stderr, "fused_gr_read_multi: invalid arguments\n");
        std::exit(1);
    }
    GrMulti m;
    for (int t = 0; t < n_tok; ++t) {
        m.a[t] = a[t];
        const FusedGrArgs& x = a[t];
        if (!x.R || !x.w_norm || !x.w_down || !x.w_up || !x.lo || !x.rs || !x.mixed || (x.w_inject && !x.inject_out) ||
            (x.apply && (!x.bo_prev || !x.inj_prev || !x.R_out)) || x.w_down != a[0].w_down || x.w_up != a[0].w_up ||
            x.w_inject != a[0].w_inject || x.w_norm != a[0].w_norm) {
            std::fprintf(stderr, "fused_gr_read_multi: invalid arguments for token %d\n", t);
            std::exit(1);
        }
    }
    m.xn = xn_scratch;
    m.part = xn_scratch + (size_t) n_tok * D;
    m.T = n_tok;
    cudaStream_t st = (cudaStream_t) stream;
    gr_norm_multi_kernel<<<dim3(n_tok, HC), THREADS, 0, st>>>(m);
    gr_down_multi_kernel<<<dim3((ROWS_D + WARPS - 1) / WARPS, KQ), THREADS, 0, st>>>(m);
    gr_up_multi_kernel<<<UPM_BLOCKS, THREADS, 0, st>>>(m);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "fused_gr_read_multi: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

size_t fused_gr_multi_scratch_floats(int n_tok) { return (size_t) n_tok * D + (size_t) KQ * n_tok * ROWS_D; }

bool fused_gr_supported(int64_t n_embd, int64_t hc, int64_t hc_lr) {
    return n_embd == N && hc == HC && hc_lr == LR;
}

void fused_gr_read(const FusedGrArgs& a, void* stream) {
    if (!a.R || !a.w_norm || !a.w_down || !a.w_up || !a.lo || !a.rs || !a.mixed ||
        (a.w_inject && !a.inject_out) || (a.apply && (!a.bo_prev || !a.inj_prev || !a.R_out)) ||
        (a.apply && a.inj_prev == a.inject_out)) {
        std::fprintf(stderr, "fused_gr_read: invalid arguments\n");
        std::exit(1);
    }
    cudaStream_t st = (cudaStream_t) stream;
    gr_down_kernel<<<DOWN_BLOCKS + 1, THREADS, 0, st>>>(a);
    gr_up_kernel<<<UP_BLOCKS, THREADS, 0, st>>>(a);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "fused_gr_read: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

void fused_gr_write(float* R, const float* bo, const float* inj, int n_tok, void* stream) {
    const int64_t n = (int64_t) n_tok * D;
    gr_write_fused_kernel<<<(unsigned) ((n + THREADS - 1) / THREADS), THREADS, 0, (cudaStream_t) stream>>>(R, bo, inj, n);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "fused_gr_write: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

}  // namespace strata::kernels
