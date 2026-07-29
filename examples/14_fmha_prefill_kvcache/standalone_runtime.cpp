#include "standalone_runtime.hpp"

#include <stdexcept>
#include <vector>

#include "standalone_profiling.hpp"

namespace sgl_standalone {
namespace {
sycl::queue* current_queue = nullptr;
void* workspace_ptr = nullptr;
std::size_t workspace_bytes = 0;
sycl::event last_kernel_event;
bool has_last_kernel_event = false;
std::vector<sycl::event> kernel_events;
}  // namespace

void set_queue(sycl::queue* q) {
  current_queue = q;
}

sycl::queue& queue() {
  if (current_queue == nullptr) {
    throw std::runtime_error("standalone SYCL queue was not initialized");
  }
  return *current_queue;
}

void set_last_event(const sycl::event& event) {
  last_kernel_event = event;
  has_last_kernel_event = true;
  kernel_events.push_back(event);
}

sycl::event last_event() {
  if (!has_last_kernel_event) {
    throw std::runtime_error("standalone FMHA kernel event was not recorded");
  }
  return last_kernel_event;
}

void clear_events() {
  kernel_events.clear();
}

const std::vector<sycl::event>& recorded_events() {
  return kernel_events;
}

void* workspace(std::size_t bytes) {
  if (bytes == 0) {
    return nullptr;
  }
  auto& q = queue();
  if (bytes > workspace_bytes) {
    q.wait();
    if (workspace_ptr != nullptr) {
      sycl::free(workspace_ptr, q);
    }
    workspace_ptr = sycl::malloc_device(bytes, q);
    if (workspace_ptr == nullptr) {
      throw std::runtime_error("failed to allocate standalone FMHA workspace");
    }
    workspace_bytes = bytes;
  }
  return workspace_ptr;
}

void release_workspace() {
  if (workspace_ptr != nullptr && current_queue != nullptr) {
    current_queue->wait();
    sycl::free(workspace_ptr, *current_queue);
  }
  workspace_ptr = nullptr;
  workspace_bytes = 0;
  has_last_kernel_event = false;
  kernel_events.clear();
}

}  // namespace sgl_standalone
