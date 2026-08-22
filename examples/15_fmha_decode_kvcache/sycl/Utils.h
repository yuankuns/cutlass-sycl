#pragma once

#include <sycl/sycl.hpp>

// The upstream SGLang header also exposes ATen/XPU queue helpers.  This
// standalone CUTLASS example only needs the work-group fence used by FMHA.
static inline void barrier() {
  sycl::group_barrier(
      sycl::ext::oneapi::this_work_item::get_work_group<3>(),
      sycl::memory_scope::work_group);
}
