// src/kernels/cuda/native_mma.cu - see include/strata/kernels/native_mma.hpp.
//
// A block owns 16 weight rows; lane (g = lane / 4, t = lane % 4) holds rows g and g + 8 of the mma A fragment and
// activation column g of its B fragment. Every format decodes 64-value units of a row pair into signed bytes; the
// int32 products are exact, and each unit adds its float contribution in a fixed order, so the result of a column is
// independent of ncols. The block layouts and codebooks are llama.cpp's ggml-common.h (MIT, third_party/ggml).
#include "strata/kernels/native_mma.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {

struct Q81Block {
    half2 ds;
    int8_t qs[32];
};

constexpr int NWT = 8;   // warps per block; they share each 16-row tile's 64-value units round robin

__device__ __forceinline__ void mma16(int (&c)[4], int a0, int a1, int b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a0), "r"(a1), "r"(b));
}
__device__ __forceinline__ void mma32(int (&c)[4], int a0, int a1, int a2, int a3, int b0, int b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ int ld32a(const uint8_t* p) { return *reinterpret_cast<const int*>(p); }
__device__ __forceinline__ int ld32u(const uint8_t* p) {
    const uint16_t* q = reinterpret_cast<const uint16_t*>(p);
    return int(q[0]) | (int(q[1]) << 16);
}
__device__ __forceinline__ int ld32(const uint8_t* p, bool al) { return al ? ld32a(p) : ld32u(p); }
__device__ __forceinline__ int sub32(int v) { return int(((uint32_t(v) | 0x80808080u) - 0x20202020u) ^ 0x80808080u); }
// Word t of the 16 bytes at p (2-byte aligned), for lane t of a quad: the quad reads a row's bytes once, contiguously,
// and shares them by shuffles (the fragment layout otherwise reads 8 rows per load, a few bytes each).
__device__ __forceinline__ uint32_t word_at(const uint8_t* p) {   // 4 bytes at a 2-byte aligned address
    const uint16_t* q = reinterpret_cast<const uint16_t*>(p);
    return (reinterpret_cast<uintptr_t>(p) & 3) == 0 ? *reinterpret_cast<const uint32_t*>(q)
                                                     : uint32_t(q[0]) | (uint32_t(q[1]) << 16);
}
__device__ __forceinline__ uint32_t quad_word(const uint8_t* p, int t) { return word_at(p + 4 * t); }
__device__ __forceinline__ float h2f(const uint8_t* p) { return __half2float(*reinterpret_cast<const half*>(p)); }

// The lane's activation views: B column g; C columns 2t and 2t + 1.  A column past ncols reads as zero.
struct Cols {
    const Q81Block* xg; const Q81Block* x0; const Q81Block* x1;
    bool gok, ok0, ok1;
    __device__ int b(int kb, int i) const { return gok ? reinterpret_cast<const int*>(xg[kb].qs)[i] : 0; }
    __device__ float d0(int kb) const { return ok0 ? __low2float(x0[kb].ds) : 0.f; }
    __device__ float d1(int kb) const { return ok1 ? __low2float(x1[kb].ds) : 0.f; }
};

// A format F decodes 64-value units of a row pair (unit), reading its codebook (TBL words; tword, gtable) from
// shared memory or through L1 (SMEM).
// C fragment: c[0] (row g, col 2t), c[1] (row g, col 2t + 1), c[2] (row g + 8, col 2t), c[3] (row g + 8, col 2t + 1).

struct Q6K {   // 210-byte blocks: ql[128] qh[64] scales[16] d; a unit is chunks j and j + 2 of half h
    static constexpr int QK = 256, BYTES = 210, TBL = 0;
    static constexpr bool SMEM = false;
    __device__ static uint32_t tword(int) { return 0; }
    __device__ static void unit(const uint8_t* w0, const uint8_t* w1, int u, int t, const Cols& c, float (&acc)[4],
                                const uint32_t*) {
        const int sb = u >> 2, h = (u >> 1) & 1, j = u & 1;
        const uint8_t* b0 = w0 + size_t(sb) * BYTES;
        const uint8_t* b1 = w1 + size_t(sb) * BYTES;
        const bool al = (reinterpret_cast<uintptr_t>(b0) & 3) == 0;   // rows g and g + 8 share it
        int ql0[2], ql1[2], qh0[2], qh1[2];
#pragma unroll
        for (int s = 0; s < 2; ++s) {
            ql0[s] = ld32(b0 + 64 * h + 32 * j + 16 * s + 4 * t, al);
            ql1[s] = ld32(b1 + 64 * h + 32 * j + 16 * s + 4 * t, al);
            qh0[s] = ld32(b0 + 128 + 32 * h + 16 * s + 4 * t, al);
            qh1[s] = ld32(b1 + 128 + 32 * h + 16 * s + 4 * t, al);
        }
        const int scA0 = *reinterpret_cast<const uint16_t*>(b0 + 192 + 8 * h + 2 * j);
        const int scB0 = *reinterpret_cast<const uint16_t*>(b0 + 196 + 8 * h + 2 * j);
        const int scA1 = *reinterpret_cast<const uint16_t*>(b1 + 192 + 8 * h + 2 * j);
        const int scB1 = *reinterpret_cast<const uint16_t*>(b1 + 196 + 8 * h + 2 * j);
        const float d0 = h2f(b0 + 208), d1 = h2f(b1 + 208);
        float s[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int k = 0; k < 2; ++k) {          // chunk cc = j + 2k: nibble k, qh shift 2cc
            const int cc = j + 2 * k, kb = sb * 8 + 4 * h + cc;
            int C[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};
#pragma unroll
            for (int sh = 0; sh < 2; ++sh) {
                const int a0 = sub32(((ql0[sh] >> (4 * k)) & 0x0F0F0F0F) | (((qh0[sh] >> (2 * cc)) & 0x03030303) << 4));
                const int a1 = sub32(((ql1[sh] >> (4 * k)) & 0x0F0F0F0F) | (((qh1[sh] >> (2 * cc)) & 0x03030303) << 4));
                mma16(C[sh], a0, a1, c.b(kb, 4 * sh + t));
            }
            const int w0s = k ? scB0 : scA0, w1s = k ? scB1 : scA1;
            const int sa0 = int8_t(w0s), sb0 = int8_t(w0s >> 8), sa1 = int8_t(w1s), sb1 = int8_t(w1s >> 8);
            const float x0 = c.d0(kb), x1 = c.d1(kb);
            s[0] += x0 * float(sa0 * C[0][0] + sb0 * C[1][0]);
            s[1] += x1 * float(sa0 * C[0][1] + sb0 * C[1][1]);
            s[2] += x0 * float(sa1 * C[0][2] + sb1 * C[1][2]);
            s[3] += x1 * float(sa1 * C[0][3] + sb1 * C[1][3]);
        }
        acc[0] += d0 * s[0]; acc[1] += d0 * s[1]; acc[2] += d1 * s[2]; acc[3] += d1 * s[3];
    }
};

__device__ __forceinline__ void scale_min_k4(int j, const uint8_t* q, int& sc, int& m) {
    if (j < 4) { sc = q[j] & 63; m = q[j + 4] & 63; }
    else { sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}

// Q4_K (144 bytes: dm scales[12] qs[128]) and Q5_K (176 bytes: dm scales[12] qh[32] qs[128]): a unit is sub-blocks
// 2i (low nibbles) and 2i + 1 (high nibbles) of qs[32i .. 32i + 32)
template<bool Q5>
struct QK45 {
    static constexpr int QK = 256, BYTES = Q5 ? 176 : 144, QS = Q5 ? 48 : 16, TBL = 0;
    static constexpr bool SMEM = false;
    __device__ static uint32_t tword(int) { return 0; }
    __device__ static void unit(const uint8_t* w0, const uint8_t* w1, int u, int t, const Cols& c, float (&acc)[4],
                                const uint32_t*) {
        const int sb = u >> 2, i = u & 3;
        const uint8_t* b0 = w0 + size_t(sb) * BYTES;
        const uint8_t* b1 = w1 + size_t(sb) * BYTES;
        int q0[2], q1[2], h0[2] = {0, 0}, h1[2] = {0, 0};
#pragma unroll
        for (int s = 0; s < 2; ++s) {
            q0[s] = ld32a(b0 + QS + 32 * i + 16 * s + 4 * t);
            q1[s] = ld32a(b1 + QS + 32 * i + 16 * s + 4 * t);
            if (Q5) { h0[s] = ld32a(b0 + 16 + 16 * s + 4 * t); h1[s] = ld32a(b1 + 16 + 16 * s + 4 * t); }
        }
        const half2 dm0 = *reinterpret_cast<const half2*>(b0), dm1 = *reinterpret_cast<const half2*>(b1);
        float sd[4] = {0.f, 0.f, 0.f, 0.f}, sm[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int k = 0; k < 2; ++k) {          // sub-block 2i + k
            const int sub = 2 * i + k, kb = sb * 8 + sub;
            int a[4];
#pragma unroll
            for (int s = 0; s < 2; ++s) {
                int v0 = (q0[s] >> (4 * k)) & 0x0F0F0F0F, v1 = (q1[s] >> (4 * k)) & 0x0F0F0F0F;
                if (Q5) { v0 |= ((h0[s] >> sub) & 0x01010101) << 4; v1 |= ((h1[s] >> sub) & 0x01010101) << 4; }
                a[2 * s] = v0; a[2 * s + 1] = v1;
            }
            const int bb0 = c.b(kb, t), bb1 = c.b(kb, 4 + t);
            int C[4] = {0, 0, 0, 0}, S[4] = {0, 0, 0, 0};
            mma32(C, a[0], a[1], a[2], a[3], bb0, bb1);
            mma32(S, 0x01010101, 0x01010101, 0x01010101, 0x01010101, bb0, bb1);
            int sc0, m0, sc1, m1;
            scale_min_k4(sub, b0 + 4, sc0, m0);
            scale_min_k4(sub, b1 + 4, sc1, m1);
            const float x0 = c.d0(kb), x1 = c.d1(kb);
            sd[0] += x0 * float(sc0 * C[0]); sm[0] += x0 * float(m0 * S[0]);
            sd[1] += x1 * float(sc0 * C[1]); sm[1] += x1 * float(m0 * S[1]);
            sd[2] += x0 * float(sc1 * C[2]); sm[2] += x0 * float(m1 * S[2]);
            sd[3] += x1 * float(sc1 * C[3]); sm[3] += x1 * float(m1 * S[3]);
        }
        const float2 f0 = __half22float2(dm0), f1 = __half22float2(dm1);
        acc[0] += f0.x * sd[0] - f0.y * sm[0]; acc[1] += f0.x * sd[1] - f0.y * sm[1];
        acc[2] += f1.x * sd[2] - f1.y * sm[2]; acc[3] += f1.x * sd[3] - f1.y * sm[3];
    }
};

__device__ __forceinline__ int2 iq4_lookup(int q4) {
    const uint32_t* table32 = (const uint32_t*) kvalues_iq4nl;
    uint32_t tmp[2];
    const uint32_t sel = 0x32103210 | ((q4 & 0x88888888) >> 1);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const uint32_t shift = 16 * i;
        const uint32_t low = __byte_perm(table32[0], table32[1], q4 >> shift);
        const uint32_t high = __byte_perm(table32[2], table32[3], q4 >> shift);
        tmp[i] = __byte_perm(low, high, sel >> shift);
    }
    return make_int2(__byte_perm(tmp[0], tmp[1], 0x6420), __byte_perm(tmp[0], tmp[1], 0x7531));
}

struct IQ4XS {   // 136 bytes: d scales_h scales_l[4] qs[128]; sub-block j: qs[16j..16j+16), low nibble -> values 0..15
    static constexpr int QK = 256, BYTES = 136, TBL = 0;
    static constexpr bool SMEM = false;
    __device__ static uint32_t tword(int) { return 0; }
    __device__ static void unit(const uint8_t* w0, const uint8_t* w1, int u, int t, const Cols& c, float (&acc)[4],
                                const uint32_t*) {
        const int sb = u >> 2, i = u & 3;
        const uint8_t* b0 = w0 + size_t(sb) * BYTES;
        const uint8_t* b1 = w1 + size_t(sb) * BYTES;
        const int sh0 = *reinterpret_cast<const uint16_t*>(b0 + 2), sh1 = *reinterpret_cast<const uint16_t*>(b1 + 2);
        const int sl0 = b0[4 + i], sl1 = b1[4 + i];
        float s[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            const int sub = 2 * i + k, kb = sb * 8 + sub;
            const int2 v0 = iq4_lookup(ld32a(b0 + 8 + 16 * sub + 4 * t));
            const int2 v1 = iq4_lookup(ld32a(b1 + 8 + 16 * sub + 4 * t));
            int C[4] = {0, 0, 0, 0};
            mma32(C, v0.x, v1.x, v0.y, v1.y, c.b(kb, t), c.b(kb, 4 + t));
            const int ls0 = (((sl0 >> (4 * k)) & 0xF) | (((sh0 >> (2 * sub)) & 3) << 4)) - 32;
            const int ls1 = (((sl1 >> (4 * k)) & 0xF) | (((sh1 >> (2 * sub)) & 3) << 4)) - 32;
            const float x0 = c.d0(kb), x1 = c.d1(kb);
            s[0] += x0 * float(ls0 * C[0]); s[1] += x1 * float(ls0 * C[1]);
            s[2] += x0 * float(ls1 * C[2]); s[3] += x1 * float(ls1 * C[3]);
        }
        const float d0 = h2f(b0), d1 = h2f(b1);
        acc[0] += d0 * s[0]; acc[1] += d0 * s[1]; acc[2] += d1 * s[2]; acc[3] += d1 * s[3];
    }
};

// Sign bits b0..b3 -> bytes 0x01 (spread) and 0xFF masks; the i-quant grids hold positive bytes, so
// (grid ^ mask) + spread negates the marked bytes without carries between them.
__device__ __forceinline__ int neg4(uint32_t grid, uint32_t bits4) {
    const uint32_t one = ((bits4 & 0xF) * 0x00204081u) & 0x01010101u;
    return int((grid ^ (one * 0xFFu)) + one);
}
__device__ __forceinline__ uint32_t ksign8(uint32_t v7) {   // 7 sign bits and their parity as bit 7 (llama.cpp ksigns)
    v7 &= 0x7F;
    return v7 | ((__popc(v7) & 1) << 7);
}

// IQ3_S, 110 bytes: d qs[64] qh[8] signs[32] scales[4].  Sub-block j holds grid entries e = 0..7 (4 values each):
// index qs[8j + e] | qh[j] bit e << 8, signs byte 4j + e / 2, nibble e & 1.  Lane t takes entries t and 4 + t.
struct IQ3S {
    static constexpr int QK = 256, BYTES = 110, TBL = 512;
    static constexpr bool SMEM = true;
    __device__ static uint32_t tword(int i) { return iq3s_grid[i]; }
    __device__ static const uint32_t* gtable() { return iq3s_grid; }
    __device__ static void unit(const uint8_t* w0, const uint8_t* w1, int u, int t, const Cols& c, float (&acc)[4],
                                const uint32_t* grid) {
        const int sb = u >> 2, i = u & 3, base = (threadIdx.x & 31) & ~3;
        const uint8_t* b[2] = {w0 + size_t(sb) * BYTES, w1 + size_t(sb) * BYTES};
        uint32_t q[2], g[2];
        int qh[2], sc[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            q[r] = quad_word(b[r] + 2 + 16 * i, t);                          // indices of sub-blocks 2i, 2i + 1
            g[r] = t < 2 ? word_at(b[r] + 74 + 8 * i + 4 * t) : 0u;         // their sign nibbles
            qh[r] = *reinterpret_cast<const uint16_t*>(b[r] + 66 + 2 * i);
            sc[r] = b[r][106 + i];
        }
        float s[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            const int kb = sb * 8 + 2 * i + k;
            int a[4], l[2];
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                const uint32_t ia = __shfl_sync(0xffffffffu, q[r], base | (2 * k));       // entries 0..3
                const uint32_t ib = __shfl_sync(0xffffffffu, q[r], base | (2 * k + 1));   // entries 4..7
                const uint32_t sw = __shfl_sync(0xffffffffu, g[r], base | k);
                const int h = qh[r] >> (8 * k);
                a[r] = neg4(grid[((ia >> (8 * t)) & 0xFF) | (((h >> t) & 1) << 8)], (sw >> (4 * t)) & 0xF);
                a[2 + r] = neg4(grid[((ib >> (8 * t)) & 0xFF) | (((h >> (4 + t)) & 1) << 8)], (sw >> (16 + 4 * t)) & 0xF);
                l[r] = 1 + 2 * ((sc[r] >> (4 * k)) & 0xF);
            }
            int C[4] = {0, 0, 0, 0};
            mma32(C, a[0], a[1], a[2], a[3], c.b(kb, t), c.b(kb, 4 + t));
            const float x0 = c.d0(kb), x1 = c.d1(kb);
            s[0] += x0 * float(l[0] * C[0]); s[1] += x1 * float(l[0] * C[1]);
            s[2] += x0 * float(l[1] * C[2]); s[3] += x1 * float(l[1] * C[3]);
        }
        const float d0 = h2f(b[0]), d1 = h2f(b[1]);
        acc[0] += d0 * s[0]; acc[1] += d0 * s[1]; acc[2] += d1 * s[2]; acc[3] += d1 * s[3];
    }
};

struct IQ4NL {   // 18 bytes: d qs[16], 32 values; a unit is two blocks
    static constexpr int QK = 32, BYTES = 18, TBL = 0;
    static constexpr bool SMEM = false;
    __device__ static uint32_t tword(int) { return 0; }
    __device__ static void unit(const uint8_t* w0, const uint8_t* w1, int u, int t, const Cols& c, float (&acc)[4],
                                const uint32_t*) {
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            const int kb = 2 * u + k;
            const uint8_t* b0 = w0 + size_t(kb) * BYTES;
            const uint8_t* b1 = w1 + size_t(kb) * BYTES;
            const int2 v0 = iq4_lookup(ld32u(b0 + 2 + 4 * t)), v1 = iq4_lookup(ld32u(b1 + 2 + 4 * t));
            int C[4] = {0, 0, 0, 0};
            mma32(C, v0.x, v1.x, v0.y, v1.y, c.b(kb, t), c.b(kb, 4 + t));
            const float d0 = h2f(b0), d1 = h2f(b1), x0 = c.d0(kb), x1 = c.d1(kb);
            acc[0] += d0 * x0 * float(C[0]); acc[1] += d0 * x1 * float(C[1]);
            acc[2] += d1 * x0 * float(C[2]); acc[3] += d1 * x1 * float(C[3]);
        }
    }
};

struct Q20 {   // 18 bytes: d qs[16], 64 values of 2 bits; lane t takes values 8t..8t+7 of each 32
    static constexpr int QK = 64, BYTES = 18, TBL = 0;
    static constexpr bool SMEM = false;
    __device__ static uint32_t tword(int) { return 0; }
    __device__ static void unit(const uint8_t* w0, const uint8_t* w1, int u, int t, const Cols& c, float (&acc)[4],
                                const uint32_t*) {
        const uint8_t* b0 = w0 + size_t(u) * BYTES;
        const uint8_t* b1 = w1 + size_t(u) * BYTES;
        const float d0 = h2f(b0), d1 = h2f(b1);
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            const int kb = 2 * u + k;
            int a[4];
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                const int q = *reinterpret_cast<const int16_t*>((r ? b1 : b0) + 2 + 8 * k + 2 * t);
                const int qe = __byte_perm(0x020100FF, 0x020100FF, q >> 0);
                const int qo = __byte_perm(0x020100FF, 0x020100FF, q >> 2);
                a[r] = __byte_perm(qe, qo, 0x5140);
                a[2 + r] = __byte_perm(qe, qo, 0x7362);
            }
            int C[4] = {0, 0, 0, 0};
            mma32(C, a[0], a[1], a[2], a[3], c.b(kb, 2 * t), c.b(kb, 2 * t + 1));
            const float x0 = c.d0(kb), x1 = c.d1(kb);
            acc[0] += d0 * x0 * float(C[0]); acc[1] += d0 * x1 * float(C[1]);
            acc[2] += d1 * x0 * float(C[2]); acc[3] += d1 * x1 * float(C[3]);
        }
    }
};

// IQ2_XXS, 66 bytes: d, then per sub-block j 8 bytes at 2 + 8j: 4 grid indices (8 values each), then 4 x 7 sign bits
// and a 4-bit scale.  Lane t takes grid entry t: values 8t..8t+7.
struct IQ2XXS {
    static constexpr int QK = 256, BYTES = 66, TBL = 512;
    static constexpr bool SMEM = false;
    __device__ static uint32_t tword(int i) { return reinterpret_cast<const uint32_t*>(iq2xxs_grid)[i]; }
    __device__ static const uint32_t* gtable() { return reinterpret_cast<const uint32_t*>(iq2xxs_grid); }
    __device__ static void unit(const uint8_t* w0, const uint8_t* w1, int u, int t, const Cols& c, float (&acc)[4],
                                const uint32_t* tbl) {
        const int sb = u >> 2, i = u & 3, base = (threadIdx.x & 31) & ~3;
        const uint8_t* b0 = w0 + size_t(sb) * BYTES;
        const uint8_t* b1 = w1 + size_t(sb) * BYTES;
        const uint2* grid = reinterpret_cast<const uint2*>(tbl);
        // the unit's 16 bytes of a row (sub-blocks 2i, 2i + 1: indices, then signs and scale), word t in lane t
        const uint32_t q[2] = {quad_word(b0 + 2 + 16 * i, t), quad_word(b1 + 2 + 16 * i, t)};
        float s[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            const int sub = 2 * i + k, kb = sb * 8 + sub;
            int a[4], ls[2];
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                const uint32_t idx = __shfl_sync(0xffffffffu, q[r], base | (2 * k));
                const uint32_t aux = __shfl_sync(0xffffffffu, q[r], base | (2 * k + 1));
                const uint2 gp = grid[(idx >> (8 * t)) & 0xFF];
                const uint32_t sg = ksign8(aux >> (7 * t));
                a[r] = neg4(gp.x, sg);
                a[2 + r] = neg4(gp.y, sg >> 4);
                ls[r] = int((aux >> 27) | 1);
            }
            int C[4] = {0, 0, 0, 0};
            mma32(C, a[0], a[1], a[2], a[3], c.b(kb, 2 * t), c.b(kb, 2 * t + 1));
            const float x0 = c.d0(kb), x1 = c.d1(kb);
            s[0] += x0 * float(ls[0] * C[0]); s[1] += x1 * float(ls[0] * C[1]);
            s[2] += x0 * float(ls[1] * C[2]); s[3] += x1 * float(ls[1] * C[3]);
        }
        const float d0 = 0.125f * h2f(b0), d1 = 0.125f * h2f(b1);
        acc[0] += d0 * s[0]; acc[1] += d0 * s[1]; acc[2] += d1 * s[2]; acc[3] += d1 * s[3];
    }
};

// IQ3_XXS, 98 bytes: d, grid indices qs[64] (4 values each), then per sub-block j a u32 at 66 + 4j: 4 x 7 sign bits
// and a 4-bit scale.  Lane t takes entries 2t and 2t + 1: values 8t..8t+7.
struct IQ3XXS {
    static constexpr int QK = 256, BYTES = 98, TBL = 256;
    static constexpr bool SMEM = false;
    __device__ static uint32_t tword(int i) { return iq3xxs_grid[i]; }
    __device__ static const uint32_t* gtable() { return iq3xxs_grid; }
    __device__ static void unit(const uint8_t* w0, const uint8_t* w1, int u, int t, const Cols& c, float (&acc)[4],
                                const uint32_t* tbl) {
        const int sb = u >> 2, i = u & 3;
        const uint8_t* b0 = w0 + size_t(sb) * BYTES;
        const uint8_t* b1 = w1 + size_t(sb) * BYTES;
        float s[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            const int sub = 2 * i + k, kb = sb * 8 + sub;
            int a[4], ls[2];
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                const uint8_t* b = r ? b1 : b0;
                const uint32_t aux = uint32_t(ld32u(b + 66 + 4 * sub));
                const int idx = *reinterpret_cast<const uint16_t*>(b + 2 + 8 * sub + 2 * t);
                const uint32_t sg = ksign8(aux >> (7 * t));
                a[r] = neg4(tbl[idx & 0xFF], sg);
                a[2 + r] = neg4(tbl[idx >> 8], sg >> 4);
                ls[r] = int(2 * (aux >> 28) + 1);
            }
            int C[4] = {0, 0, 0, 0};
            mma32(C, a[0], a[1], a[2], a[3], c.b(kb, 2 * t), c.b(kb, 2 * t + 1));
            const float x0 = c.d0(kb), x1 = c.d1(kb);
            s[0] += x0 * float(ls[0] * C[0]); s[1] += x1 * float(ls[0] * C[1]);
            s[2] += x0 * float(ls[1] * C[2]); s[3] += x1 * float(ls[1] * C[3]);
        }
        const float d0 = 0.25f * h2f(b0), d1 = 0.25f * h2f(b1);
        acc[0] += d0 * s[0]; acc[1] += d0 * s[1]; acc[2] += d1 * s[2]; acc[3] += d1 * s[3];
    }
};

// IQ2_XS (74 bytes: d, qs[32] u16 = 9-bit grid index | 7 sign bits << 9, scales[8]) and IQ2_S (82 bytes: d, grid index
// low bytes qs[32], sign bytes[32], qh[8] with 2 index bits per entry, scales[8]).  Sub-block j is entries 4j..4j+3;
// scale nibble 0 covers entries 0, 1 and nibble 1 entries 2, 3, so lanes t < 2 and t >= 2 take separate products.
// Lane t takes entry t: values 8t..8t+7.
template<bool S>
struct IQ2XSS {
    static constexpr int QK = 256, BYTES = S ? 82 : 74, TBL = S ? 2048 : 1024, SC = S ? 74 : 66;
    static constexpr bool SMEM = !S;
    __device__ static uint32_t tword(int i) {
        return S ? reinterpret_cast<const uint32_t*>(iq2s_grid)[i] : reinterpret_cast<const uint32_t*>(iq2xs_grid)[i];
    }
    __device__ static const uint32_t* gtable() {
        return S ? reinterpret_cast<const uint32_t*>(iq2s_grid) : reinterpret_cast<const uint32_t*>(iq2xs_grid);
    }
    __device__ static void unit(const uint8_t* w0, const uint8_t* w1, int u, int t, const Cols& c, float (&acc)[4],
                                const uint32_t* tbl) {
        const int sb = u >> 2, i = u & 3;
        const uint8_t* b0 = w0 + size_t(sb) * BYTES;
        const uint8_t* b1 = w1 + size_t(sb) * BYTES;
        const uint2* grid = reinterpret_cast<const uint2*>(tbl);
        const bool lo = t < 2;
        float s[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            const int sub = 2 * i + k, kb = sb * 8 + sub;
            int a[4], l0[2], l1[2];
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                const uint8_t* b = r ? b1 : b0;
                uint2 gp;
                uint32_t sg;
                if (S) {
                    gp = grid[b[2 + 4 * sub + t] | (((b[66 + sub] >> (2 * t)) & 3) << 8)];
                    sg = b[34 + 4 * sub + t];
                } else {
                    const uint32_t q = *reinterpret_cast<const uint16_t*>(b + 2 + 8 * sub + 2 * t);
                    gp = grid[q & 0x1FF];
                    sg = ksign8(q >> 9);
                }
                a[r] = neg4(gp.x, sg);
                a[2 + r] = neg4(gp.y, sg >> 4);
                const int sc = b[SC + sub];
                l0[r] = 2 * (sc & 0xF) + 1;
                l1[r] = 2 * (sc >> 4) + 1;
            }
            int CA[4] = {0, 0, 0, 0}, CB[4] = {0, 0, 0, 0};
            const int u0 = c.b(kb, 2 * t), u1 = c.b(kb, 2 * t + 1);
            mma32(CA, lo ? a[0] : 0, lo ? a[1] : 0, lo ? a[2] : 0, lo ? a[3] : 0, u0, u1);
            mma32(CB, lo ? 0 : a[0], lo ? 0 : a[1], lo ? 0 : a[2], lo ? 0 : a[3], u0, u1);
            const float x0 = c.d0(kb), x1 = c.d1(kb);
            s[0] += x0 * float(l0[0] * CA[0] + l1[0] * CB[0]); s[1] += x1 * float(l0[0] * CA[1] + l1[0] * CB[1]);
            s[2] += x0 * float(l0[1] * CA[2] + l1[1] * CB[2]); s[3] += x1 * float(l0[1] * CA[3] + l1[1] * CB[3]);
        }
        const float d0 = 0.125f * h2f(b0), d1 = 0.125f * h2f(b1);
        acc[0] += d0 * s[0]; acc[1] += d0 * s[1]; acc[2] += d1 * s[2]; acc[3] += d1 * s[3];
    }
};

// The codebook the units read: a block's shared copy (F::SMEM, TBL words) or the global table through L1, whichever
// measured faster for the format.
template<typename F>
__device__ __forceinline__ const uint32_t* load_table(uint32_t* tbl) {
    if constexpr (F::TBL > 0) {
        if constexpr (F::SMEM) {
            for (int i = threadIdx.x; i < F::TBL; i += blockDim.x) tbl[i] = F::tword(i);
            __syncthreads();
            return tbl;
        } else {
            return F::gtable();
        }
    }
    return tbl;
}

template<typename F>
__global__ void __launch_bounds__(NWT * 32) kernel(const uint8_t* __restrict__ w, const Q81Block* __restrict__ x,
                                                  float* __restrict__ y, int n_in, int n_out, int ncols) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int xs = n_in / 32, nu = n_in / 64;
    const size_t row_bytes = size_t(n_in / F::QK) * F::BYTES;
    const int r0 = blockIdx.x * 16 + g, r1 = r0 + 8;
    const uint8_t* w0 = w + size_t(min(r0, n_out - 1)) * row_bytes;
    const uint8_t* w1 = w + size_t(min(r1, n_out - 1)) * row_bytes;
    Cols c;
    c.gok = g < ncols; c.ok0 = 2 * t < ncols; c.ok1 = 2 * t + 1 < ncols;
    c.xg = x + size_t(c.gok ? g : 0) * xs;
    c.x0 = x + size_t(c.ok0 ? 2 * t : 0) * xs;
    c.x1 = x + size_t(c.ok1 ? 2 * t + 1 : 0) * xs;
    __shared__ uint32_t tbl[F::SMEM ? F::TBL : 1];
    const uint32_t* cb = load_table<F>(tbl);
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int u = warp; u < nu; u += NWT) F::unit(w0, w1, u, t, c, acc, cb);
    __shared__ float part[NWT - 1][4][32];
    if (warp > 0)
        for (int i = 0; i < 4; ++i) part[warp - 1][i][lane] = acc[i];
    __syncthreads();
    if (warp > 0) return;
    for (int wv = 0; wv < NWT - 1; ++wv)
        for (int i = 0; i < 4; ++i) acc[i] += part[wv][i][lane];
    const int rr[4] = {r0, r0, r1, r1}, cc4[4] = {2 * t, 2 * t + 1, 2 * t, 2 * t + 1};
    for (int i = 0; i < 4; ++i)
        if (rr[i] < n_out && cc4[i] < ncols) y[size_t(cc4[i]) * n_out + rr[i]] = acc[i];
}

// ---------------------------------------------------------------- grouped routed experts
// Group g (blockIdx.y) is one expert with its entries e0..e1 (at most 8: a window's tokens), taken as the columns.
// Gate/up: a tile is 8 rows of gate (the fragment's rows g) and the same 8 rows of up (rows g + 8), so each lane holds
// both halves of its outputs and writes silu(gate) * up.  Down: a tile is 16 rows.  A block owns one tile of one
// group; its warps share the tile's units round robin and warp 0 sums their partials in warp order.
constexpr int GU_NW = 4;   // n_embd = 2560: 40 units per tile
constexpr int DN_NW = 2;   // n_ff = 640: 10 units per tile

__device__ __forceinline__ Cols entry_cols(const Q81Block* x, const int32_t* rows, int e0, int n, int xs, int g, int t) {
    Cols c;
    c.gok = g < n; c.ok0 = 2 * t < n; c.ok1 = 2 * t + 1 < n;
    const int e = c.gok ? g : 0, e_0 = c.ok0 ? 2 * t : 0, e_1 = c.ok1 ? 2 * t + 1 : 0;
    c.xg = x + size_t(rows ? rows[e0 + e] : e0 + e) * xs;
    c.x0 = x + size_t(rows ? rows[e0 + e_0] : e0 + e_0) * xs;
    c.x1 = x + size_t(rows ? rows[e0 + e_1] : e0 + e_1) * xs;
    return c;
}

template<int NW>
__device__ __forceinline__ bool reduce_warps(float (&acc)[4], int warp, int lane) {
    __shared__ float part[NW - 1][4][32];
    if (warp > 0)
        for (int i = 0; i < 4; ++i) part[warp - 1][i][lane] = acc[i];
    __syncthreads();
    if (warp > 0) return false;
    for (int wv = 0; wv < NW - 1; ++wv)
        for (int i = 0; i < 4; ++i) acc[i] += part[wv][i][lane];
    return true;
}

template<typename F>
__global__ void __launch_bounds__(GU_NW * 32) expert_gu_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                               const int32_t* __restrict__ grp_start,
                                                               const int32_t* __restrict__ n_groups,
                                                               const int32_t* __restrict__ ent_tok,
                                                               const Q81Block* __restrict__ xq, NativeExpertLayout L,
                                                               float* __restrict__ h) {
    const int grp = blockIdx.y;
    if (grp >= *n_groups) return;
    __shared__ uint32_t tbl[F::SMEM ? F::TBL : 1];
    const uint32_t* cb = load_table<F>(tbl);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int n_ff = int(L.n_ff), nu = int(L.n_embd / 64), r = blockIdx.x * 8 + g;   // rows g: gate r; rows g + 8: up r
    const int e0 = grp_start[grp], n = grp_start[grp + 1] - e0;
    const Cols c = entry_cols(xq, ent_tok, e0, n, int(L.n_embd / 32), g, t);
    const uint8_t* blob = reinterpret_cast<const uint8_t*>(grp_ptr[grp]);
    const uint8_t* wg = blob + size_t(r) * L.gu_row;
    const uint8_t* wu = blob + L.up_off + size_t(r) * L.gu_row;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int u = warp; u < nu; u += GU_NW) F::unit(wg, wu, u, t, c, acc, cb);
    if (!reduce_warps<GU_NW>(acc, warp, lane)) return;
    if (c.ok0) h[size_t(e0 + 2 * t) * n_ff + r] = acc[0] / (1.0f + __expf(-acc[0])) * acc[2];
    if (c.ok1) h[size_t(e0 + 2 * t + 1) * n_ff + r] = acc[1] / (1.0f + __expf(-acc[1])) * acc[3];
}

template<typename F>
__global__ void __launch_bounds__(DN_NW * 32) expert_down_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                                 const int32_t* __restrict__ grp_start,
                                                                 const int32_t* __restrict__ n_groups,
                                                                 const int32_t* __restrict__ ent_dst,
                                                                 const Q81Block* __restrict__ hq, NativeExpertLayout L,
                                                                 float* __restrict__ out) {
    const int grp = blockIdx.y;
    if (grp >= *n_groups) return;
    __shared__ uint32_t tbl[F::SMEM ? F::TBL : 1];
    const uint32_t* cb = load_table<F>(tbl);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int n_embd = int(L.n_embd), nu = int(L.n_ff / 64), r0 = blockIdx.x * 16 + g, r1 = r0 + 8;
    const int e0 = grp_start[grp], n = grp_start[grp + 1] - e0;
    const Cols c = entry_cols(hq, nullptr, e0, n, int(L.n_ff / 32), g, t);
    const uint8_t* blob = reinterpret_cast<const uint8_t*>(grp_ptr[grp]) + L.down_off;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int u = warp; u < nu; u += DN_NW) F::unit(blob + size_t(r0) * L.d_row, blob + size_t(r1) * L.d_row, u, t, c, acc, cb);
    if (!reduce_warps<DN_NW>(acc, warp, lane)) return;
    if (c.ok0) {
        float* o = out + size_t(ent_dst[e0 + 2 * t]) * n_embd;
        o[r0] = acc[0];
        o[r1] = acc[2];
    }
    if (c.ok1) {
        float* o = out + size_t(ent_dst[e0 + 2 * t + 1]) * n_embd;
        o[r0] = acc[1];
        o[r1] = acc[3];
    }
}

template<typename F>
void launch(const void* w, const void* x, float* y, int n_in, int n_out, int ncols, cudaStream_t s) {
    kernel<F><<<unsigned((n_out + 15) / 16), NWT * 32, 0, s>>>(static_cast<const uint8_t*>(w),
                                                                static_cast<const Q81Block*>(x), y, n_in, n_out, ncols);
}

int block_values(int type) {
    switch (type) {
        case 12: case 13: case 14: case 16: case 17: case 18: case 21: case 22: case 23: return 256;
        case 42: return 64;
        case 20: return 32;
        default: return 0;
    }
}

}  // namespace

bool native_mma_supported(int ggml_type, int n_in) {
    if (ggml_type == 16 || ggml_type == 17 || ggml_type == 18 || ggml_type == 22) return false;   // expert formats only
    const int qk = block_values(ggml_type);
    return qk > 0 && n_in > 0 && n_in % 64 == 0 && n_in % qk == 0;
}

void native_mma(int ggml_type, const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols,
                void* stream) {
    if (!native_mma_supported(ggml_type, n_in) || n_out <= 0 || ncols < 1 || ncols > 8)
        throw std::invalid_argument("native_mma: type " + std::to_string(ggml_type) + ", n_in " + std::to_string(n_in) +
                                    ", n_out " + std::to_string(n_out) + ", ncols " + std::to_string(ncols));
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    switch (ggml_type) {
        case 12: launch<QK45<false>>(weights, x_q8_1, y, n_in, n_out, ncols, s); break;
        case 13: launch<QK45<true>>(weights, x_q8_1, y, n_in, n_out, ncols, s); break;
        case 14: launch<Q6K>(weights, x_q8_1, y, n_in, n_out, ncols, s); break;
        case 20: launch<IQ4NL>(weights, x_q8_1, y, n_in, n_out, ncols, s); break;
        case 21: launch<IQ3S>(weights, x_q8_1, y, n_in, n_out, ncols, s); break;
        case 23: launch<IQ4XS>(weights, x_q8_1, y, n_in, n_out, ncols, s); break;
        case 42: launch<Q20>(weights, x_q8_1, y, n_in, n_out, ncols, s); break;
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("native_mma launch: ") + cudaGetErrorString(e));
}

namespace {
bool fits(int type, int64_t n) {
    const int qk = block_values(type);
    return qk > 0 && n > 0 && n % 64 == 0 && n % qk == 0;
}
template<typename F>
void launch_gu(const NativeExpertLayout& L, const unsigned long long* gp, const int32_t* gs, const int32_t* gn,
               const int32_t* tok, int64_t cap_groups, const void* x, float* h, cudaStream_t s) {
    const dim3 grid(unsigned(L.n_ff / 8), unsigned(cap_groups));
    expert_gu_kernel<F><<<grid, GU_NW * 32, 0, s>>>(gp, gs, gn, tok, static_cast<const Q81Block*>(x), L, h);
}
template<typename F>
void launch_down(const NativeExpertLayout& L, const unsigned long long* gp, const int32_t* gs, const int32_t* gn,
                 const int32_t* dst, int64_t cap_groups, const void* hq, float* out, cudaStream_t s) {
    const dim3 grid(unsigned(L.n_embd / 16), unsigned(cap_groups));
    expert_down_kernel<F><<<grid, DN_NW * 32, 0, s>>>(gp, gs, gn, dst, static_cast<const Q81Block*>(hq), L, out);
}
}  // namespace

bool native_mma_experts_supported(int gu_type, int d_type, int64_t n_embd, int64_t n_ff) {
    const bool gu = gu_type == 16 || gu_type == 17 || gu_type == 18 || gu_type == 21 || gu_type == 22 || gu_type == 23 ||
                    gu_type == 42;
    const bool dn = d_type == 20 || d_type == 23 || d_type == 42;
    return gu && dn && fits(gu_type, n_embd) && fits(d_type, n_ff) && n_embd % 16 == 0;
}

void native_mma_experts(const NativeExpertLayout& L, const unsigned long long* grp_ptr, const int32_t* grp_start,
                        const int32_t* n_groups, const int32_t* ent_dst, const int32_t* ent_tok, int64_t cap_groups,
                        int64_t cap_entries, const void* x_q8_1, float* h, void* hq, float* out, void* stream) {
    if (!native_mma_experts_supported(L.gu_type, L.d_type, L.n_embd, L.n_ff))
        throw std::invalid_argument("native_mma_experts: gate/up type " + std::to_string(L.gu_type) + ", down type " +
                                    std::to_string(L.d_type));
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    switch (L.gu_type) {
        case 16: launch_gu<IQ2XXS>(L, grp_ptr, grp_start, n_groups, ent_tok, cap_groups, x_q8_1, h, s); break;
        case 17: launch_gu<IQ2XSS<false>>(L, grp_ptr, grp_start, n_groups, ent_tok, cap_groups, x_q8_1, h, s); break;
        case 18: launch_gu<IQ3XXS>(L, grp_ptr, grp_start, n_groups, ent_tok, cap_groups, x_q8_1, h, s); break;
        case 21: launch_gu<IQ3S>(L, grp_ptr, grp_start, n_groups, ent_tok, cap_groups, x_q8_1, h, s); break;
        case 22: launch_gu<IQ2XSS<true>>(L, grp_ptr, grp_start, n_groups, ent_tok, cap_groups, x_q8_1, h, s); break;
        case 23: launch_gu<IQ4XS>(L, grp_ptr, grp_start, n_groups, ent_tok, cap_groups, x_q8_1, h, s); break;
        case 42: launch_gu<Q20>(L, grp_ptr, grp_start, n_groups, ent_tok, cap_groups, x_q8_1, h, s); break;
    }
    quantize_q8_1_rows(h, cap_entries, L.n_ff, hq, stream);
    switch (L.d_type) {
        case 20: launch_down<IQ4NL>(L, grp_ptr, grp_start, n_groups, ent_dst, cap_groups, hq, out, s); break;
        case 23: launch_down<IQ4XS>(L, grp_ptr, grp_start, n_groups, ent_dst, cap_groups, hq, out, s); break;
        case 42: launch_down<Q20>(L, grp_ptr, grp_start, n_groups, ent_dst, cap_groups, hq, out, s); break;
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("native_mma_experts launch: ") + cudaGetErrorString(e));
}

}  // namespace strata::kernels
