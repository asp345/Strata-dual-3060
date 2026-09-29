// include/strata/kernels/fused_gr.hpp - plan v0.3 P3: the hyper-connection read in TWO kernels, with the
// previous half's write folded in.
//
// The native path spends six kernels per `gr_read` (norm, down MMVF, silu, up MMVF, gate+mean, inject MMVF) and
// one per `gr_write`, 96 + 96 times per token, and its up projection (10240 rows of 320) runs one 160-thread
// block per row: 20.6 us for 6.5 MB.  Here:
//
//   fused_gr_down : R' = R + bo_prev * 2 sigmoid(inj_prev / hc)  (only when `apply`, computed on the fly)
//                   rs[c] = rsqrt(mean(R'[c]^2) + eps),  xn = R' * w_norm * rs
//                   lo[k] = silu((w_down[k] . xn) / hc)          k < hc_lr
//                   inject[c] = w_inject[c] . xn                 when w_inject is given
//   fused_gr_up   : R <- R' in place for this block's columns (when `apply`)
//                   mixed[d] = mean_c  xn[c,d] * sigmoid(w_up[c*n_embd + d] . lo)
//
// FP32 activations and BF16 weights, like the native MMVF contract; the summation order differs from it (G-C
// judges the result).  Geometry is the artifact's: n_embd 2560, hc 4, hc_lr 320.  `inj_prev` and `inject_out`
// must be different buffers (every block reads the former while one block writes the latter).
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels {

struct FusedGrArgs {
    const float* R = nullptr;          ///< (hc, n_embd), read by `down`; `up` updates it in place when apply
    float* R_out = nullptr;            ///< == R for the in-place update
    bool apply = false;                ///< fold the previous half's gr_write
    const float* bo_prev = nullptr;    ///< that half's block output, n_embd
    const float* inj_prev = nullptr;   ///< that half's injection, hc
    const float* w_norm = nullptr;     ///< (hc * n_embd) f32
    const uint16_t* w_down = nullptr;  ///< bf16 [hc_lr][hc*n_embd]
    const uint16_t* w_up = nullptr;    ///< bf16 [hc*n_embd][hc_lr]
    const uint16_t* w_inject = nullptr;///< bf16 [hc][hc*n_embd], or null (the final mixer)
    float eps = 1e-6f;
    float* lo = nullptr;               ///< workspace, hc_lr floats
    float* rs = nullptr;               ///< workspace, hc floats
    float* inject_out = nullptr;       ///< hc floats (when w_inject)
    float* mixed = nullptr;            ///< n_embd
};

bool fused_gr_supported(int64_t n_embd, int64_t hc, int64_t hc_lr);
void fused_gr_read(const FusedGrArgs& a, void* stream);

/// Plan v0.3 P6: the same read for up to 8 tokens that share the weights (a verify window): the weights are read
/// once for all of them.  `a[t]` is token t's arguments (its own R, pending write, lo, rs, inject, mixed; the four
/// weight pointers and eps must be the same for every t); `xn_scratch` holds fused_gr_multi_scratch_floats(n_tok)
/// floats.  A token's outputs do not depend on n_tok; they round differently from `fused_gr_read` (the down
/// projection is summed over four quarters of its inputs).
constexpr int kFusedGrMaxT = 8;
size_t fused_gr_multi_scratch_floats(int n_tok);
void fused_gr_read_multi(const FusedGrArgs* a, int n_tok, float* xn_scratch, void* stream);
/// The folded write alone, `R <- R + bo * 2 sigmoid(inj / hc)` with the arithmetic `fused_gr_read` applies it with,
/// for `n_tok` tokens (R hc * n_embd, bo n_embd, inj hc per token): a pipeline stage's last layer, whose write has no
/// next read on the same GPU to fold into.  A read with `apply = false` after it sees what a folding read computes.
void fused_gr_write(float* R, const float* bo, const float* inj, int n_tok, void* stream);

}  // namespace strata::kernels
