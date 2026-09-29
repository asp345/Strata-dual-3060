#pragma once

#include <cstdint>

namespace strata::kernels {

// Dense GGUF projections (Q4_K, Q5_K, Q6_K, IQ4_XS, IQ3_S, IQ4_NL, Q2_0) against 1..8 Q8_1 activation columns on
// the int8 tensor cores (mma.sync m16n8k32 / m16n8k16): each weight block is decoded once for all columns, so the
// cost barely grows with ncols. A column's result depends only on its own activations, never on ncols or on its
// position among the columns. Layout as native_mmvq (native_mmvq.hpp); n_in must be a multiple of 64.
bool native_mma_supported(int ggml_type, int n_in);
void native_mma(int ggml_type, const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols,
                void* stream);

struct NativeExpertLayout;   // iq_kernels.hpp

// Grouped routed experts on the same tensor cores (native_expert_grouped's contract, iq_kernels.hpp): a group's
// entries (at most 8) are the columns. Gate/up write silu(gate) * up to h (cap_entries x n_ff), quantized to hq, then
// down writes out.
bool native_mma_experts_supported(int gu_type, int d_type, int64_t n_embd, int64_t n_ff);
void native_mma_experts(const NativeExpertLayout& L, const unsigned long long* grp_ptr, const int32_t* grp_start,
                        const int32_t* n_groups, const int32_t* ent_dst, const int32_t* ent_tok, int64_t cap_groups,
                        int64_t cap_entries, const void* x_q8_1, float* h, void* hq, float* out, void* stream);

}  // namespace strata::kernels
