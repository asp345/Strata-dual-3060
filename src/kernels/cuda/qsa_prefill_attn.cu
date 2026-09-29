// src/kernels/cuda/qsa_prefill_attn.cu - see include/strata/kernels/qsa_decode_attn.hpp (qsa_prefill_attn).
//
// One block per (query, KV head): the 12 query heads of the KV head are the rows of an fp16 mma (m16n8k16, fp32
// accumulators, rows 12-15 zero), the selected cells are walked 32 at a time with an online softmax.  A chunk's K and
// V rows are dequantized into shared memory as fp16; S = Q K^T and O += P V run on the tensor cores.
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/kv_q8.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int HD = 256;
constexpr int G = 12;              // query heads per KV head
constexpr int C = 32;              // cells per chunk
constexpr int WARPS = 4;
constexpr int THREADS = WARPS * 32;
constexpr int LD = HD + 8;         // shared row stride in halves: rows 16 bytes apart in bank terms
constexpr int PLD = C + 8;

__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void* p) {
    const uint32_t a = (uint32_t) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void ldsm_x2(uint32_t (&r)[2], const void* p) {
    const uint32_t a = (uint32_t) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(r[0]), "=r"(r[1]) : "r"(a));
}
__device__ __forceinline__ void ldsm_x2_t(uint32_t (&r)[2], const void* p) {
    const uint32_t a = (uint32_t) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];" : "=r"(r[0]), "=r"(r[1]) : "r"(a));
}
__device__ __forceinline__ void mma(float (&c)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// One thread's share of a chunk: 64 dimensions [64 * seg, +64) of one cell's K and V rows, loaded raw (int8 codes and
// their group scale, or fp16), then written to shared memory as fp16.
template <bool Q8>
struct Raw {
    uint4 k[Q8 ? 4 : 8], v[Q8 ? 4 : 8];
    float ks = 0.f, vs = 0.f;
    bool valid = false;
};
template <bool Q8>
__device__ __forceinline__ void load_raw(const QsaAttnPools& p, long long row, int seg, Raw<Q8>& r) {
    r.valid = row >= 0;
    if (!r.valid) return;
    if constexpr (Q8) {
        const uint4* kc = reinterpret_cast<const uint4*>(p.k_q + row * HD + seg * 64);
        const uint4* vc = reinterpret_cast<const uint4*>(p.v_q + row * HD + seg * 64);
#pragma unroll
        for (int i = 0; i < 4; ++i) { r.k[i] = kc[i]; r.v[i] = vc[i]; }
        r.ks = __half2float(__ushort_as_half(p.k_scale[row * (HD / KV_Q8_GROUP) + seg]));
        r.vs = __half2float(__ushort_as_half(p.v_scale[row * (HD / KV_Q8_GROUP) + seg]));
    } else {
        const uint4* kc = reinterpret_cast<const uint4*>(p.k_pool + row * HD + seg * 64);
        const uint4* vc = reinterpret_cast<const uint4*>(p.v_pool + row * HD + seg * 64);
#pragma unroll
        for (int i = 0; i < 8; ++i) { r.k[i] = kc[i]; r.v[i] = vc[i]; }
    }
}
template <bool Q8>
__device__ __forceinline__ void store_half(const uint4* src, float sc, bool valid, __half* dst) {
    if (!valid) {
#pragma unroll
        for (int i = 0; i < 8; ++i) reinterpret_cast<uint4*>(dst)[i] = make_uint4(0, 0, 0, 0);
        return;
    }
    if constexpr (Q8) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int8_t* c8 = reinterpret_cast<const int8_t*>(&src[i]);
            __half2 h[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) h[j] = __floats2half2_rn((float) c8[2 * j] * sc, (float) c8[2 * j + 1] * sc);
            reinterpret_cast<uint4*>(dst)[2 * i] = *reinterpret_cast<const uint4*>(&h[0]);
            reinterpret_cast<uint4*>(dst)[2 * i + 1] = *reinterpret_cast<const uint4*>(&h[4]);
        }
    } else {
#pragma unroll
        for (int i = 0; i < 8; ++i) reinterpret_cast<uint4*>(dst)[i] = src[i];
    }
}

template <bool Q8>
__global__ void __launch_bounds__(THREADS) prefill_attn_kernel(const float* __restrict__ q, QsaAttnPools p,
                                                               const int32_t* __restrict__ ids,
                                                               const int32_t* __restrict__ steps, int cap,
                                                               int n_kv_heads, int page_size, float scale,
                                                               float* __restrict__ attn) {
    __shared__ __align__(16) __half Qs[16][LD];
    __shared__ __align__(16) __half Ks[C][LD];
    __shared__ __align__(16) __half Vs[C][LD];
    __shared__ __align__(16) __half Ps[16][PLD];
    __shared__ float red[WARPS][16];
    const int qi = blockIdx.x, kvh = blockIdx.y;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5, g = lane >> 2, tq = lane & 3;
    const int n_ids = __ldg(steps + (size_t) qi * kStepCount + kStepWidth);
    const int32_t* sel = ids + (size_t) qi * cap;
    const float* qrow = q + ((size_t) qi * n_kv_heads * G + (size_t) kvh * G) * HD;
    for (int i = t; i < 16 * HD / 2; i += THREADS) {
        const int r = i / (HD / 2), c2 = i % (HD / 2);
        const float2 v = r < G ? *reinterpret_cast<const float2*>(qrow + (size_t) r * HD + 2 * c2) : make_float2(0.f, 0.f);
        *reinterpret_cast<__half2*>(&Qs[r][2 * c2]) = __floats2half2_rn(v.x, v.y);
    }
    // running max and sum of rows g and g + 8 (every thread of a row agrees: they compute the same values)
    float m_run[2] = {-FLT_MAX, -FLT_MAX}, l_run[2] = {0.f, 0.f};
    float o[8][4];
#pragma unroll
    for (int j = 0; j < 8; ++j) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.f;
    // thread t stages 64 dims (segment t % 4) of the chunk's cell t / 4; the next chunk's rows are loaded while this one
    // is computed
    const int my_cell = t >> 2, seg = t & 3;
    auto row_of = [&](int c) -> long long {
        if (c >= n_ids) return -1;
        const int cell = sel[c];
        return ((long long) p.page_table[cell / page_size] * n_kv_heads + kvh) * page_size + cell % page_size;
    };
    Raw<Q8> raw;
    load_raw<Q8>(p, row_of(my_cell), seg, raw);
    for (int c0 = 0; c0 < n_ids; c0 += C) {
        const int n_here = min(C, n_ids - c0);
        __syncthreads();                                   // the previous chunk's K, V and P are consumed
        store_half<Q8>(raw.k, raw.ks, raw.valid, &Ks[my_cell][seg * 64]);
        store_half<Q8>(raw.v, raw.vs, raw.valid, &Vs[my_cell][seg * 64]);
        if (c0 + C < n_ids) load_raw<Q8>(p, row_of(c0 + C + my_cell), seg, raw);
        __syncthreads();
        // S for cells [8 * warp, +8): 16 k-steps over the head dimension
        float s[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int k0 = 0; k0 < HD; k0 += 16) {
            uint32_t a[4], b[2];
            ldsm_x4(a, &Qs[lane & 15][k0 + (lane >> 4) * 8]);
            ldsm_x2(b, &Ks[warp * 8 + (lane & 7)][k0 + ((lane >> 3) & 1) * 8]);
            mma(s, a, b);
        }
        // s[0], s[1]: row g, cells 8 * warp + 2 tq, + 1; s[2], s[3]: row g + 8
        const int cl = warp * 8 + 2 * tq;
#pragma unroll
        for (int i = 0; i < 4; ++i) s[i] = (cl + (i & 1) < n_here) ? s[i] * scale : -FLT_MAX;
        float mx[2] = {fmaxf(s[0], s[1]), fmaxf(s[2], s[3])};
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 1));
            mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 2));
        }
        if (tq == 0) { red[warp][g] = mx[0]; red[warp][g + 8] = mx[1]; }
        __syncthreads();
        float m_new[2], alpha[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            float m = m_run[r];
#pragma unroll
            for (int w = 0; w < WARPS; ++w) m = fmaxf(m, red[w][g + 8 * r]);
            m_new[r] = m;
            alpha[r] = __expf(m_run[r] - m);
        }
        const float p0 = __expf(s[0] - m_new[0]), p1 = __expf(s[1] - m_new[0]);
        const float p2 = __expf(s[2] - m_new[1]), p3 = __expf(s[3] - m_new[1]);
        *reinterpret_cast<__half2*>(&Ps[g][cl]) = __floats2half2_rn(p0, p1);
        *reinterpret_cast<__half2*>(&Ps[g + 8][cl]) = __floats2half2_rn(p2, p3);
        float sm[2] = {p0 + p1, p2 + p3};
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            sm[r] += __shfl_xor_sync(0xffffffffu, sm[r], 1);
            sm[r] += __shfl_xor_sync(0xffffffffu, sm[r], 2);
        }
        __syncthreads();                                   // every warp has read red (maxima); Ps is written
        if (tq == 0) { red[warp][g] = sm[0]; red[warp][g + 8] = sm[1]; }
        __syncthreads();
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            float l = 0.f;
#pragma unroll
            for (int w = 0; w < WARPS; ++w) l += red[w][g + 8 * r];
            l_run[r] = l_run[r] * alpha[r] + l;
            m_run[r] = m_new[r];
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            o[j][0] *= alpha[0]; o[j][1] *= alpha[0];
            o[j][2] *= alpha[1]; o[j][3] *= alpha[1];
        }
        // O[:, 64 * warp + 8 j ..] += P V over the chunk's 32 cells (2 k-steps)
#pragma unroll
        for (int k0 = 0; k0 < C; k0 += 16) {
            uint32_t a[4];
            ldsm_x4(a, &Ps[lane & 15][k0 + (lane >> 4) * 8]);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                uint32_t b[2];
                ldsm_x2_t(b, &Vs[k0 + (lane & 15)][warp * 64 + 8 * j]);
                mma(o[j], a, b);
            }
        }
    }
    // o[j]: rows g, g + 8; dims 64 * warp + 8 j + 2 tq, + 1
    float* out = attn + ((size_t) qi * n_kv_heads * G + (size_t) kvh * G) * HD;
    const float i0 = l_run[0] > 0.f ? 1.f / l_run[0] : 0.f, i1 = l_run[1] > 0.f ? 1.f / l_run[1] : 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int d = warp * 64 + 8 * j + 2 * tq;
        if (g < G) *reinterpret_cast<float2*>(out + (size_t) g * HD + d) = make_float2(o[j][0] * i0, o[j][1] * i0);
        if (g + 8 < G) *reinterpret_cast<float2*>(out + (size_t) (g + 8) * HD + d) = make_float2(o[j][2] * i1, o[j][3] * i1);
    }
}

}  // namespace

bool qsa_prefill_attn_supported(const QsaAttnPools& pools, const QsaShapes& s) {
    return pools.k_q4 == nullptr && s.head_dim == HD && s.n_head == (int64_t) G * s.n_head_kv;
}

void qsa_prefill_attn(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps, int64_t cap,
                      const QsaShapes& s, float* attn, int64_t n_q, void* stream) {
    if (n_q <= 0) return;
    if (!qsa_prefill_attn_supported(pools, s) || !ids || !steps || !pools.page_table || cap <= 0) {
        std::fprintf(stderr, "qsa_prefill_attn: unsupported geometry or missing buffers\n");
        std::exit(1);
    }
    const dim3 grid((unsigned) n_q, (unsigned) s.n_head_kv);
    const float scale = 1.0f / sqrtf((float) HD);
    cudaStream_t st = (cudaStream_t) stream;
    if (pools.k_q != nullptr)
        prefill_attn_kernel<true><<<grid, THREADS, 0, st>>>(q, pools, ids, steps, (int) cap, (int) s.n_head_kv,
                                                            (int) s.page_size, scale, attn);
    else
        prefill_attn_kernel<false><<<grid, THREADS, 0, st>>>(q, pools, ids, steps, (int) cap, (int) s.n_head_kv,
                                                             (int) s.page_size, scale, attn);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa_prefill_attn: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

}  // namespace strata::kernels
