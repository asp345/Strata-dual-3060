// src/kernels/cuda/qsa_select.cu - see include/strata/kernels/qsa_select.hpp.
#include "strata/kernels/qsa_select.hpp"

#include <cuda_runtime.h>

#include <cfloat>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int IDX_DIM = 128, IDX_HEADS = 4, R = 4;
constexpr int SCORE_WARPS = 8;
constexpr int TOPK_T = 256;

__device__ __forceinline__ uint32_t order_key(float s) {
    const float v = s + 0.0f;
    if (!(v == v)) return 0u;
    const uint32_t b = __float_as_uint(v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

__global__ void __launch_bounds__(SCORE_WARPS * 32) block_scores_kernel(const float* __restrict__ pooled,
                                                                        const float* __restrict__ dead,
                                                                        const float* __restrict__ q_idx,
                                                                        const int32_t* __restrict__ steps,
                                                                        int64_t max_blocks, float* __restrict__ out) {
    const int64_t qi = blockIdx.y;
    const int32_t* st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid];
    const int64_t b = (int64_t) blockIdx.x * SCORE_WARPS + (threadIdx.x >> 5);
    if (b > n_bid || b >= max_blocks) return;
    const int lane = threadIdx.x & 31;
    const float* key = (b == n_bid) ? dead : pooled + b * IDX_DIM;
    const float4 k4 = *reinterpret_cast<const float4*>(key + lane * 4);
    const float* q = q_idx + qi * IDX_HEADS * IDX_DIM + lane * 4;
    float score = 0.0f;
#pragma unroll
    for (int h = 0; h < IDX_HEADS; ++h) {
        const float4 q4 = *reinterpret_cast<const float4*>(q + h * IDX_DIM);
        float d = k4.x * q4.x + k4.y * q4.y + k4.z * q4.z + k4.w * q4.w;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
        score += d > 0.0f ? d : 0.0f;
    }
    if (lane == 0) {
        if (b == n_bid && n_kv % R != 0) score += 1e9f;
        out[qi * max_blocks + b] = score;
    }
}

// qsa_block_scores for many queries (a prompt's): a tile of ST_Q queries x ST_B blocks per thread block, the query and
// key slices staged in shared memory (dimension-major, so a warp's reads do not collide) and each thread's 4 x 8
// dot products kept in registers - no shuffles.  The dots are summed in dimension order, so a score can differ from
// the warp kernel's in the last bits.  Blocks past a query's n_bid are not written; n_bid itself is written by
// block_scores_dead_kernel.
constexpr int ST_Q = 128, ST_B = 128, ST_D = 32;
__global__ void __launch_bounds__(256) block_scores_tiled_kernel(const float* __restrict__ pooled,
                                                                 const float* __restrict__ q_idx,
                                                                 const int32_t* __restrict__ steps, int64_t nq,
                                                                 int64_t max_blocks, float* __restrict__ out) {
    __shared__ __align__(16) float qs[ST_D][ST_Q];
    __shared__ __align__(16) float ks[ST_D][ST_B];
    const int64_t q0 = (int64_t) blockIdx.y * ST_Q, b0 = (int64_t) blockIdx.x * ST_B;
    const int64_t q_last = min(q0 + ST_Q, nq) - 1;
    const int64_t nb_hi = steps[q_last * kStepCount + kStepNBid];   // the tile's largest n_bid (positions ascend)
    if (b0 >= nb_hi) return;
    const int t = threadIdx.x, tx = t & 15, ty = t >> 4;         // 16 x 16 threads, each 8 blocks x 8 queries
    float score[8][8] = {};
    for (int h = 0; h < IDX_HEADS; ++h) {
        float acc[8][8] = {};
        for (int d0 = 0; d0 < IDX_DIM; d0 += ST_D) {
            __syncthreads();
            for (int i = t; i < ST_Q * ST_D; i += 256) {
                const int qq = i / ST_D, dd = i % ST_D;
                const int64_t q = q0 + qq;
                qs[dd][qq] = q < nq ? q_idx[(q * IDX_HEADS + h) * IDX_DIM + d0 + dd] : 0.0f;
            }
            for (int i = t; i < ST_B * ST_D; i += 256) {
                const int bb = i / ST_D, dd = i % ST_D;
                const int64_t b = b0 + bb;
                ks[dd][bb] = b < nb_hi ? pooled[b * IDX_DIM + d0 + dd] : 0.0f;
            }
            __syncthreads();
#pragma unroll 8
            for (int dd = 0; dd < ST_D; ++dd) {
                const float4 qv = *reinterpret_cast<const float4*>(&qs[dd][ty * 4]);
                const float4 qw = *reinterpret_cast<const float4*>(&qs[dd][64 + ty * 4]);
                const float4 kv = *reinterpret_cast<const float4*>(&ks[dd][tx * 4]);
                const float4 kw = *reinterpret_cast<const float4*>(&ks[dd][64 + tx * 4]);
                const float qa[8] = {qv.x, qv.y, qv.z, qv.w, qw.x, qw.y, qw.z, qw.w};
                const float ka[8] = {kv.x, kv.y, kv.z, kv.w, kw.x, kw.y, kw.z, kw.w};
#pragma unroll
                for (int i = 0; i < 8; ++i)
#pragma unroll
                    for (int j = 0; j < 8; ++j) acc[i][j] = fmaf(qa[i], ka[j], acc[i][j]);
            }
        }
#pragma unroll
        for (int i = 0; i < 8; ++i)
#pragma unroll
            for (int j = 0; j < 8; ++j) score[i][j] += acc[i][j] > 0.0f ? acc[i][j] : 0.0f;
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int64_t q = q0 + (i < 4 ? ty * 4 + i : 64 + ty * 4 + i - 4);
        if (q >= nq) continue;
        const int64_t n_bid = steps[q * kStepCount + kStepNBid];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int64_t b = b0 + (j < 4 ? tx * 4 + j : 64 + tx * 4 + j - 4);
            if (b < n_bid && b < max_blocks) out[q * max_blocks + b] = score[i][j];
        }
    }
}

// a query's tail block n_bid: the `dead` key, +1e9 when the block has cells (as block_scores_kernel)
__global__ void __launch_bounds__(32) block_scores_dead_kernel(const float* __restrict__ dead,
                                                               const float* __restrict__ q_idx,
                                                               const int32_t* __restrict__ steps, int64_t max_blocks,
                                                               float* __restrict__ out) {
    const int64_t qi = blockIdx.x;
    const int32_t* st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid];
    if (n_bid >= max_blocks) return;
    const int lane = threadIdx.x;
    const float4 k4 = *reinterpret_cast<const float4*>(dead + lane * 4);
    const float* q = q_idx + qi * IDX_HEADS * IDX_DIM + lane * 4;
    float score = 0.0f;
#pragma unroll
    for (int h = 0; h < IDX_HEADS; ++h) {
        const float4 q4 = *reinterpret_cast<const float4*>(q + h * IDX_DIM);
        float d = k4.x * q4.x + k4.y * q4.y + k4.z * q4.z + k4.w * q4.w;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
        score += d > 0.0f ? d : 0.0f;
    }
    if (lane == 0) out[qi * max_blocks + n_bid] = score + (n_kv % R != 0 ? 1e9f : 0.0f);
}

__global__ void __launch_bounds__(TOPK_T) block_topk_kernel(const float* __restrict__ scores,
                                                            const int32_t* __restrict__ steps, int64_t max_blocks,
                                                            int64_t cap, int32_t* __restrict__ ids) {
    __shared__ int hist[256];
    __shared__ int s_a[TOPK_T], s_b[TOPK_T];
    __shared__ int s_digit, s_above;
    const int64_t qi = blockIdx.x;
    const int32_t* st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid], width = st[kStepWidth];
    int32_t* out = ids + qi * cap;
    const int t = threadIdx.x;
    if (n_kv <= width) {                               // everything is selected: the identity, ascending
        for (int64_t j = t; j < n_kv; j += TOPK_T) out[j] = (int32_t) j;
        return;
    }
    const float* sc = scores + qi * max_blocks;
    const int64_t nb = n_bid + 1;                      // blocks 0..n_bid, the last possibly empty
    const int64_t per = (nb + TOPK_T - 1) / TOPK_T;
    const int64_t b0 = (int64_t) t * per, b1 = (b0 + per < nb) ? b0 + per : nb;
    auto weight = [&](int64_t b) -> int { return b < n_bid ? R : (int) (n_kv - n_bid * R); };
    // ---- radix select: the largest key thr with (cells with key >= thr) >= width, 8 bits at a time
    uint32_t prefix = 0;
    int above = 0;                                     // cells strictly above the digits fixed so far
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = t; i < 256; i += TOPK_T) hist[i] = 0;
        __syncthreads();
        const uint32_t hi_mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
        for (int64_t b = b0; b < b1; ++b) {
            const int w = weight(b);
            if (w == 0) continue;
            const uint32_t k = order_key(sc[b]);
            if ((k & hi_mask) == (prefix & hi_mask)) atomicAdd(&hist[(k >> shift) & 255], w);
        }
        __syncthreads();
        if (t == 0) {
            int cum = above, d = 255;
            for (; d > 0; --d) {
                if (cum + hist[d] >= width) break;
                cum += hist[d];
            }
            s_digit = d;
            s_above = cum;
        }
        __syncthreads();
        prefix |= (uint32_t) s_digit << shift;
        above = s_above;
        __syncthreads();
    }
    const uint32_t thr = prefix;
    const int64_t eq_budget = width - above;          // cells equal to thr that fit, lowest index first
    // ---- per-thread counts of cells above and at the threshold, then their exclusive prefixes
    int gt = 0, eq = 0;
    for (int64_t b = b0; b < b1; ++b) {
        const int w = weight(b);
        if (w == 0) continue;
        const uint32_t k = order_key(sc[b]);
        if (k > thr) gt += w;
        else if (k == thr) eq += w;
    }
    s_a[t] = gt;
    s_b[t] = eq;
    __syncthreads();
    if (t == 0) {
        int ag = 0, ae = 0;
        for (int i = 0; i < TOPK_T; ++i) {
            const int g = s_a[i], e = s_b[i];
            s_a[i] = ag; s_b[i] = ae;
            ag += g; ae += e;
        }
    }
    __syncthreads();
    const int64_t eq_before = s_b[t];
    int64_t my_eq = eq_budget - eq_before;
    if (my_eq < 0) my_eq = 0;
    if (my_eq > eq) my_eq = eq;
    const int sel = gt + (int) my_eq;
    __syncthreads();
    s_a[t] = sel;
    __syncthreads();
    if (t == 0) {
        int a = 0;
        for (int i = 0; i < TOPK_T; ++i) { const int c = s_a[i]; s_a[i] = a; a += c; }
    }
    __syncthreads();
    int64_t wpos = s_a[t];
    int64_t eq_left = my_eq;
    for (int64_t b = b0; b < b1; ++b) {
        const int w = weight(b);
        if (w == 0) continue;
        const uint32_t k = order_key(sc[b]);
        if (k > thr) {
            for (int c = 0; c < w; ++c) out[wpos++] = (int32_t) (b * R + c);
        } else if (k == thr) {
            for (int c = 0; c < w && eq_left > 0; ++c, --eq_left) out[wpos++] = (int32_t) (b * R + c);
        }
    }
}

}  // namespace

void qsa_block_scores(const float* pooled, const float* dead, const float* q_idx, const int32_t* steps, int64_t nq,
                      int64_t max_blocks, const QsaShapes& s, float* scores, void* stream, int64_t active_blocks) {
    if (nq <= 0) return;
    if (s.idx_dim != IDX_DIM || s.idx_n_head != IDX_HEADS || s.idx_block != R || nq > 65535) {
        std::fprintf(stderr, "qsa_block_scores: unsupported indexer geometry\n");
        std::exit(1);
    }
    // a block past a query's n_bid returns at once: the grid need only reach the batch's largest n_bid (C-1)
    const int64_t reach = active_blocks > 0 && active_blocks < max_blocks ? active_blocks : max_blocks;
    const dim3 grid((unsigned) ((reach + SCORE_WARPS - 1) / SCORE_WARPS), (unsigned) nq);
    block_scores_kernel<<<grid, SCORE_WARPS * 32, 0, (cudaStream_t) stream>>>(pooled, dead, q_idx, steps, max_blocks,
                                                                              scores);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "qsa_block_scores: %s\n", cudaGetErrorString(e)); std::exit(1); }
}

void qsa_block_scores_tiled(const float* pooled, const float* dead, const float* q_idx, const int32_t* steps, int64_t nq,
                            int64_t max_blocks, const QsaShapes& s, float* scores, void* stream, int64_t active_blocks) {
    if (nq <= 0) return;
    if (s.idx_dim != IDX_DIM || s.idx_n_head != IDX_HEADS || s.idx_block != R || (nq + ST_Q - 1) / ST_Q > 65535) {
        std::fprintf(stderr, "qsa_block_scores_tiled: unsupported indexer geometry\n");
        std::exit(1);
    }
    const int64_t reach = std::min(active_blocks, max_blocks);
    const dim3 grid((unsigned) ((reach + ST_B - 1) / ST_B), (unsigned) ((nq + ST_Q - 1) / ST_Q));
    block_scores_tiled_kernel<<<grid, 256, 0, (cudaStream_t) stream>>>(pooled, q_idx, steps, nq, max_blocks, scores);
    block_scores_dead_kernel<<<(unsigned) nq, 32, 0, (cudaStream_t) stream>>>(dead, q_idx, steps, max_blocks, scores);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "qsa_block_scores_tiled: %s\n", cudaGetErrorString(e)); std::exit(1); }
}

void qsa_block_topk(const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks, int64_t cap,
                    const QsaShapes& s, int32_t* ids, void* stream) {
    if (nq <= 0) return;
    if (s.idx_block != R || cap < qsa_selection_width(kTopkMaxCells, s)) {
        std::fprintf(stderr, "qsa_block_topk: unsupported geometry or cap\n");
        std::exit(1);
    }
    block_topk_kernel<<<(unsigned) nq, TOPK_T, 0, (cudaStream_t) stream>>>(scores, steps, max_blocks, cap, ids);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "qsa_block_topk: %s\n", cudaGetErrorString(e)); std::exit(1); }
}

}  // namespace strata::kernels
