#pragma once

#include <cstddef>
#include <vector>

#include <sycl/sycl.hpp>

namespace sgl_standalone {

void set_queue(sycl::queue* q);
sycl::queue& queue();
void clear_events();
void set_last_event(const sycl::event& event);
sycl::event last_event();
const std::vector<sycl::event>& last_events();
void* workspace(std::size_t bytes);
void release_workspace();

}  // namespace sgl_standalone
