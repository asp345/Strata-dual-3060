// include/strata/core/placement.hpp - which GPU runs which layers (layer-split pipeline parallelism).
//
// The model runs on one or more GPUs as a pipeline of STAGES: stage s holds the layers [first(s), end(s)) - their
// dense weights, their KV and recurrent state, and VRAM expert slots for their experts only - and a token's
// residual passes through the stages in order.  The last stage also holds the output head and the MTP draft
// layer.  Stages never run the same layer, so no tensor is duplicated and the only traffic between GPUs is the
// residual at a stage boundary (hc * n_embd floats per token).
//
// One GPU is the one-stage placement, and it is the default: every caller that never sets a placement gets
// device 0 for every layer.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

struct Placement {
    std::vector<int> devices{0};          ///< CUDA ordinals, in pipeline order
    std::vector<int64_t> bounds{0, 48};   ///< stages() + 1 entries: 0, the first layer of stages 1.., n_layers

    int stages() const { return (int) devices.size(); }
    int last() const { return stages() - 1; }
    int device(int stage) const { return devices[(size_t) stage]; }
    int64_t first(int stage) const { return bounds[(size_t) stage]; }
    int64_t end(int stage) const { return bounds[(size_t) stage + 1]; }
    int stage_of(int64_t layer) const;
    /// The stage a weight belongs to: `blk.<l>.*` to layer l's, the token embedding to the first, the rest (the
    /// output head and its hyper-connection mixer) to the last.
    int stage_of_tensor(const std::string& name) const;
};

/// Set once at startup, before anything is allocated on a GPU.  Refuses a placement that does not cover
/// [0, n_layers) with non-empty stages, or that names a device twice.
bool set_placement(const Placement& p, int64_t n_layers, std::string& err);
const Placement& placement();

/// Makes `device` current for the scope and restores the previous one.
class DeviceGuard {
public:
    explicit DeviceGuard(int device);
    ~DeviceGuard();
    DeviceGuard(const DeviceGuard&) = delete;
    DeviceGuard& operator=(const DeviceGuard&) = delete;

private:
    int prev_ = 0;
};

}  // namespace strata::core
