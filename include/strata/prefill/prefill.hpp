// include/strata/prefill/prefill.hpp - plan v0.3 P5: batched prompt processing.
//
// The prompt's positions [pos0, pos0 + n) are processed in chunks of `chunk` tokens through all 48 layers, leaving
// the session state (GDN recurrence and conv state, QSA KV pools and indexer, PLE history) where the token path
// would have left it; the decode loop then continues with the next token.  Per layer: the projections are
// tensor-core GEMMs (quantized weights dequantized to FP16 on the fly, BF16 weights as they are), the recurrences
// walk the chunk inside one kernel, and the routed experts are grouped by expert: resident ones are read from the
// VRAM tier, the others streamed from the host arena through a pinned ring on a copy stream.
//
// With the layers split over GPUs (placement.hpp) each stage runs its own layers of a chunk on its GPU, with its own
// buffers, and the chunk's residual is copied to the next stage's GPU in between.
//
// Requires the native weights (`--native`): every quantized projection must carry its GGUF blocks.
#pragma once

#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/layer.hpp"
#include "strata/core/session.hpp"
#include "strata/core/weights.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace strata::prefill {

struct PrefillStats {
    int64_t tokens = 0;
    int64_t chunks = 0;
    double ms_total = 0;
    double ms_experts_host = 0;     ///< host time staging non-resident experts
    int64_t experts_streamed = 0;   ///< expert blobs copied host -> device
    int64_t experts_dma = 0;        ///< ...of which straight from the pinned arena (no CPU copy)
    int64_t experts_resident = 0;   ///< expert-layer groups served from the VRAM tier
    double ms_ple = 0;
};

class Prefill {
public:
    Prefill();
    ~Prefill();
    Prefill(const Prefill&) = delete;
    Prefill& operator=(const Prefill&) = delete;

    /// Device memory a stage's buffers are carved from: the top slots of its GPU's expert cache, lent for the
    /// prompt and refilled after it.  Null = allocate normally.
    struct Borrow { void* base = nullptr; uint64_t bytes = 0; };

    /// `host_res`: the static residency table (n_layers x n_expert, global slot or -1) or null; `slot_addr` the
    /// device address of every slot, on its layer's GPU.  `borrow`: one per pipeline stage, or empty.
    bool init(const core::WeightTable& wt, const core::ModelGeometry& g, core::SessionState& ss,
              core::ExpertSource* src, const uint64_t* slot_addr, const int32_t* host_res, int64_t chunk,
              std::string& err, const std::vector<Borrow>& borrow = {});

    /// With borrowed buffers: lay them out again for chunks of `chunk` tokens (at most `init`'s) in `borrow` (one
    /// per stage) - a request lends only the slots its prompt needs.  Called between prompts.
    bool relayout(int64_t chunk, const std::vector<Borrow>& borrow, std::string& err);
    int64_t chunk() const;

    /// The share of the streamed experts' bytes DMA-able straight from pinned RAM (1 = all).  Sizes the streamed
    /// ring (a big one only pays when the copy engine, not the host copies, is the limit); set before bytes_needed.
    static void set_pinned_share(double share);
    static double pinned_share();

    /// Device bytes `init` needs on pipeline stage `stage`'s GPU for a chunk of `chunk` tokens (what its borrowed
    /// region must hold).
    static uint64_t bytes_needed(const core::ModelGeometry& g, const core::SessionState& ss, int64_t chunk, int stage);

    /// Positions [pos0, pos0 + n) holding `tokens`; `ss.ple_prev` must be the two tokens before pos0 (oldest
    /// first, -1 for none) and is advanced to the last two of these.
    bool run(const int64_t* tokens, int64_t n, int64_t pos0, std::string& err);

    const PrefillStats& stats() const { return stats_; }

    /// Plan v0.3 P6: called after every chunk with the chunk's final multi-stream residual rows (device, on the last
    /// stage's GPU, T x hc*n_embd, valid until the next chunk) and the chunk's first position; the MTP draft layer
    /// builds its K/V from them.  The last stage's stream is synchronized before the call, which comes from that
    /// stage's thread: with more than one GPU the stages before it may be working on the next chunk.
    std::function<bool(const float* R_rows, int64_t T, int64_t pos0, std::string& err)> on_chunk;

    /// With more than one GPU: asked, in order, for the end position of every chunk; true = the first stage waits
    /// until every stage has finished that chunk and `on_chunk` returned (where the callback reads every layer's
    /// state, a conversation checkpoint).  Null = never.
    std::function<bool(int64_t end)> hold_at;

    /// Checked before every chunk: true stops the prompt early (`run` returns false with err "cancelled").
    std::function<bool()> should_stop;

    /// The vision path: HOST rows (n_embd floats) indexed by absolute position, read in place of the token
    /// embedding where non-null (an image's <|image_pad|> cells).  Null (default): every position embeds its token.
    const float* const* embd_rows = nullptr;

private:
    struct Impl;
    bool carve(Impl& m, std::size_t T, void* alloc);   // the device buffers of a chunk (prefill.cpp's Alloc)
    std::vector<std::unique_ptr<Impl>> parts_;         // one per pipeline stage
    PrefillStats stats_;
};

}  // namespace strata::prefill
