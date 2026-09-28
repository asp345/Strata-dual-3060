// src/core/placement.cpp - see include/strata/core/placement.hpp.
#include "strata/core/placement.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>

namespace strata::core {
namespace {
Placement g_placement;
}

int Placement::stage_of(int64_t layer) const {
    for (int s = 0; s + 1 < stages(); ++s)
        if (layer < bounds[(size_t) s + 1]) return s;
    return last();
}

int Placement::stage_of_tensor(const std::string& name) const {
    if (name.rfind("blk.", 0) == 0) return stage_of(std::strtoll(name.c_str() + 4, nullptr, 10));
    if (name == "token_embd.weight") return 0;
    return last();
}

bool set_placement(const Placement& p, int64_t n_layers, std::string& err) {
    if (p.devices.empty() || p.bounds.size() != p.devices.size() + 1 || p.bounds.front() != 0 ||
        p.bounds.back() != n_layers) {
        err = "the GPU placement must cover layers 0.." + std::to_string(n_layers - 1);
        return false;
    }
    for (size_t s = 0; s + 1 < p.bounds.size(); ++s)
        if (p.bounds[s] >= p.bounds[s + 1]) {
            err = "every GPU must run at least one layer";
            return false;
        }
    std::vector<int> d = p.devices;
    std::sort(d.begin(), d.end());
    if (std::adjacent_find(d.begin(), d.end()) != d.end()) {
        err = "a GPU is named twice";
        return false;
    }
    g_placement = p;
    return true;
}

const Placement& placement() { return g_placement; }

DeviceGuard::DeviceGuard(int device) {
    cudaGetDevice(&prev_);
    if (device != prev_) cudaSetDevice(device);
}

DeviceGuard::~DeviceGuard() {
    int cur = 0;
    cudaGetDevice(&cur);
    if (cur != prev_) cudaSetDevice(prev_);
}

}  // namespace strata::core
