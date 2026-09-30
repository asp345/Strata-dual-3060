// strata/kernels/rope_yarn.hpp - YaRN (llama.cpp --rope-scaling yarn): a context longer than the one the model was
// trained on.  ggml's rope_yarn, per rotated pair: the angle at freq_scale x the position (interpolation) and at the
// position itself (extrapolation) are mixed along a ramp between the correction dims - the pairs that turn many times
// over the trained context keep their angle, the slow ones are interpolated - and cos/sin are scaled by
// 1 + 0.1 ln(1 / freq_scale).  Off (freq_scale 1, ext_factor 0) every angle and value is what it was, bit for bit.
// Kernels take it as an argument, so it is set before any CUDA graph is captured.
#pragma once

#include <cstdint>

namespace strata::kernels {

struct RopeYarn {
    float freq_scale = 1.0f, ext_factor = 0.0f, attn_factor = 1.0f;
    float corr0 = 0.0f, corr1 = 0.0f;   ///< the ramp's pairs (ggml_rope_yarn_corr_dims)
};

/// YaRN over `factor` x `orig_ctx` positions for `n_rot` rotated channels at `freq_base` (ggml's beta_fast 32 and
/// beta_slow 1).
void rope_yarn_set(double factor, int64_t orig_ctx, double freq_base, int n_rot);
const RopeYarn& rope_yarn();

#if defined(__CUDACC__)
/// cos and sin of pair `pair`'s angle `theta_extrap` (the position times the pair's frequency), YaRN applied.
__device__ __forceinline__ void rope_yarn_cs(float theta_extrap, int pair, const RopeYarn& y, float& c, float& s) {
    float theta = y.freq_scale * theta_extrap;
    float mscale = y.attn_factor;
    if (y.ext_factor != 0.0f) {
        const float r = ((float) pair - y.corr0) / fmaxf(0.001f, y.corr1 - y.corr0);
        const float mix = (1.0f - fminf(1.0f, fmaxf(0.0f, r))) * y.ext_factor;
        theta = theta * (1.0f - mix) + theta_extrap * mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / y.freq_scale);
    }
    c = cosf(theta) * mscale;
    s = sinf(theta) * mscale;
}
#endif

}  // namespace strata::kernels
