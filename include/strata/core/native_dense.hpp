#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace strata::core {
class WeightTable;

// Experimental GDN/QSA/shared-expert projection overrides. Upload unchanged native GGUF
// blocks once, then attach them to the matching canonical WeightRef. Unsupported
// types retain their canonical paths. Each projection goes to the GPU of its layer's
// pipeline stage (placement.hpp). Owns one Q8_1 scratch vector per stage, shared by
// that stage's projections, so use one ordered stream per stage and keep this object
// alive until all graphs that reference it have been destroyed and synchronized.
class NativeDense {
public:
    NativeDense() = default;
    ~NativeDense();
    NativeDense(const NativeDense&) = delete;
    NativeDense& operator=(const NativeDense&) = delete;
    bool load(const std::vector<std::string>& shards, WeightTable& table, std::string& err,
              bool include_ple_key = false);
    /// Plan v0.3 P1: the canonical tensors `load` would serve natively from these shards (eligible name, supported
    /// type, 2-D) and their device bytes, read from the GGUF headers only - so the canonical arena can skip them.
    static bool served_bytes(const std::vector<std::string>& shards, bool include_ple_key,
                             std::map<std::string, uint64_t>& out, std::string& err);
    uint64_t weight_bytes() const { return bytes_; }
    size_t tensor_count() const { return weights_.size(); }

private:
    std::vector<void*> weights_;
    std::vector<void*> scratch_;   ///< one per pipeline stage
    uint64_t bytes_ = 0;
};
} // namespace strata::core
