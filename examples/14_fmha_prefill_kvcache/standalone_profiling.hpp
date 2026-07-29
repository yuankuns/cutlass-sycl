#pragma once

#include <sycl/sycl.hpp>
#include <vector>

// Host-side profiling helpers. Kept out of standalone_runtime.hpp so that
// touching them does not force a recompile of the (very slow) generated device
// kernel translation units.
namespace sgl_standalone {

// All kernel events recorded since the last clear_events(). A single prefill
// call may enqueue more than one kernel (e.g. the split score store/load
// dispatch for HEAD_DIM=512), so device-time measurement must sum over every
// kernel rather than use last_event() alone.
void clear_events();
const std::vector<sycl::event>& recorded_events();

}  // namespace sgl_standalone
