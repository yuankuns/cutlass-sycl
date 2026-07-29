#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "standalone_profiling.hpp"
#include "standalone_runtime.hpp"
#include "sycl/kernels/flash_attention_v2/xe_fmha_fwd_prefill_dispatch.hpp"

namespace {

using bf16_t = uint16_t;

struct Config {
  int batch = 4;
  int seqlen_q = 128;
  int seqlen_k = 1024;
  int past_kv = 0;
  int heads_q = 16;
  int heads_kv = 4;
  int head_dim = 64;
  int head_dim_v = 64;
  int page_size = 64;
  bool page_table_random = false;
  bool paged = true;
  bool causal = true;
  int window_left = -1;
  int window_right = -1;
  bool sink = false;
  int warmup = 5;
  int iters = 5;
  int seed = 0;
  double atol = 5e-2;
  double rtol = 5e-2;
  bool verify = true;
  bool past_kv_set = false;
  std::vector<int> past_kv_list;
  std::vector<int> seqlen_q_list;
  std::vector<int> k_new_seqlens;
  std::vector<int> cu_seqlens_k_new;
  std::vector<int> cache_seqlens_old;
};

struct LengthInfo {
  std::vector<int32_t> q_lens;
  std::vector<int32_t> k_lens;
  std::vector<int32_t> past_lens;
  std::vector<int32_t> k_new_lens;
  std::vector<int32_t> cache_lens_old;
  std::vector<int32_t> cu_q;
  std::vector<int32_t> cu_k;
  std::vector<int32_t> cu_k_new;
  std::vector<int32_t> page_table;
  int max_q = 0;
  int max_k = 0;
  int max_k_new = 0;
  int total_q = 0;
  int total_k = 0;
  int total_k_new = 0;
  int total_pages = 0;
  int page_table_stride = 0;
  bool append_kv = false;
  bool k_new_uses_cu = false;
};

uint32_t float_bits(float x) {
  uint32_t bits;
  std::memcpy(&bits, &x, sizeof(bits));
  return bits;
}

float bits_float(uint32_t bits) {
  float x;
  std::memcpy(&x, &bits, sizeof(x));
  return x;
}

bf16_t float_to_bf16(float x) {
  uint32_t bits = float_bits(x);
  const uint32_t lsb = (bits >> 16) & 1u;
  bits += 0x7fffu + lsb;
  return static_cast<bf16_t>(bits >> 16);
}

float bf16_to_float(bf16_t x) {
  return bits_float(static_cast<uint32_t>(x) << 16);
}

bool append_kv_enabled(const Config& cfg) {
  return !cfg.k_new_seqlens.empty() || !cfg.cu_seqlens_k_new.empty();
}

int round_up_headdim_standalone(int head_size) {
  if (head_size <= 64) return 64;
  if (head_size <= 96) return 96;
  if (head_size <= 128) return 128;
  if (head_size <= 192) return 192;
  if (head_size <= 256) return 256;
  if (head_size <= 512) return 512;
  return 512;
}

bool parse_bool(const std::string& v) {
  if (v == "1" || v == "true" || v == "True" || v == "yes") return true;
  if (v == "0" || v == "false" || v == "False" || v == "no") return false;
  throw std::runtime_error("invalid bool value: " + v);
}

std::vector<int> parse_int_list(const std::string& text, const char* name) {
  std::vector<int> values;
  std::stringstream ss(text);
  std::string item;
  while (std::getline(ss, item, ',')) {
    const auto first = item.find_first_not_of(" \t");
    const auto last = item.find_last_not_of(" \t");
    if (first == std::string::npos) {
      throw std::runtime_error(std::string("empty value in --") + name);
    }
    values.push_back(std::stoi(item.substr(first, last - first + 1)));
  }
  if (values.empty()) {
    throw std::runtime_error(std::string("--") + name + " must contain at least one value");
  }
  return values;
}

Config parse_args(int argc, char** argv) {
  Config cfg;
  std::unordered_map<std::string, std::string> kv;
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg == "--help" || arg == "-h") {
      std::cout
          << "Usage: fmha_prefill_kvcache [options]\n"
          << "  --batch N --seqlen-q N --seqlen-k N --heads-q N --heads-kv N\n"
          << "  --past-kv N --past-kv-list a,b,... --seqlen-q-list a,b,...\n"
          << "  --k-new-seqlens N[,..] --cu-seqlens-k-new 0,... --cache-seqlens-old N[,..]\n"
          << "  --head-dim N --head-dim-v N --paged 0|1 --page-size N\n"
          << "  --page-table-random 0|1\n"
          << "  --causal 0|1 --window-left N --window-right N --sink 0|1\n"
          << "  --warmup N --iters N --seed N --verify 0|1 --atol F --rtol F\n";
      std::exit(0);
    }
    if (arg.rfind("--", 0) != 0 || i + 1 >= argc) {
      throw std::runtime_error("expected --key value, got: " + arg);
    }
    kv[arg.substr(2)] = argv[++i];
  }

  auto get_int = [&](const char* name, int& out) {
    auto it = kv.find(name);
    if (it != kv.end()) out = std::stoi(it->second);
  };
  auto get_double = [&](const char* name, double& out) {
    auto it = kv.find(name);
    if (it != kv.end()) out = std::stod(it->second);
  };
  auto get_bool = [&](const char* name, bool& out) {
    auto it = kv.find(name);
    if (it != kv.end()) out = parse_bool(it->second);
  };
  auto get_list_alias = [&](const char* hyphen_name, const char* underscore_name, std::vector<int>& out) {
    auto hyphen_it = kv.find(hyphen_name);
    auto underscore_it = kv.find(underscore_name);
    if (hyphen_it != kv.end() && underscore_it != kv.end()) {
      throw std::runtime_error(std::string("--") + hyphen_name + " and --" + underscore_name + " are aliases");
    }
    if (hyphen_it != kv.end()) {
      out = parse_int_list(hyphen_it->second, hyphen_name);
    } else if (underscore_it != kv.end()) {
      out = parse_int_list(underscore_it->second, underscore_name);
    }
  };

  get_int("batch", cfg.batch);
  get_int("seqlen-q", cfg.seqlen_q);
  get_int("seqlen-k", cfg.seqlen_k);
  auto past_it = kv.find("past-kv");
  if (past_it != kv.end()) {
    cfg.past_kv = std::stoi(past_it->second);
    cfg.past_kv_set = true;
  }
  auto past_list_it = kv.find("past-kv-list");
  if (past_list_it != kv.end()) {
    cfg.past_kv_list = parse_int_list(past_list_it->second, "past-kv-list");
  }
  auto q_list_it = kv.find("seqlen-q-list");
  if (q_list_it != kv.end()) {
    cfg.seqlen_q_list = parse_int_list(q_list_it->second, "seqlen-q-list");
  }
  get_list_alias("k-new-seqlens", "k_new_seqlens", cfg.k_new_seqlens);
  get_list_alias("cu-seqlens-k-new", "cu_seqlens_k_new", cfg.cu_seqlens_k_new);
  get_list_alias("cache-seqlens-old", "cache_seqlens_old", cfg.cache_seqlens_old);
  get_int("heads-q", cfg.heads_q);
  get_int("heads-kv", cfg.heads_kv);
  get_int("head-dim", cfg.head_dim);
  get_int("head-dim-v", cfg.head_dim_v);
  get_int("page-size", cfg.page_size);
  get_bool("page-table-random", cfg.page_table_random);
  get_bool("paged", cfg.paged);
  get_bool("causal", cfg.causal);
  get_int("window-left", cfg.window_left);
  get_int("window-right", cfg.window_right);
  get_bool("sink", cfg.sink);
  get_int("warmup", cfg.warmup);
  get_int("iters", cfg.iters);
  get_int("seed", cfg.seed);
  get_double("atol", cfg.atol);
  get_double("rtol", cfg.rtol);
  get_bool("verify", cfg.verify);

  if (cfg.batch <= 0 || cfg.seqlen_q <= 0 || cfg.seqlen_k <= 0) {
    throw std::runtime_error("batch, seqlen_q, and seqlen_k must be positive");
  }
  if (cfg.past_kv_set && !cfg.past_kv_list.empty()) {
    throw std::runtime_error("--past-kv and --past-kv-list are mutually exclusive");
  }
  if (cfg.past_kv_set && cfg.past_kv < 0) {
    throw std::runtime_error("--past-kv must be non-negative");
  }
  if (!cfg.past_kv_list.empty() && static_cast<int>(cfg.past_kv_list.size()) != cfg.batch) {
    throw std::runtime_error("--past-kv-list must have exactly batch entries");
  }
  if (!cfg.seqlen_q_list.empty() && static_cast<int>(cfg.seqlen_q_list.size()) != cfg.batch) {
    throw std::runtime_error("--seqlen-q-list must have exactly batch entries");
  }
  if (!cfg.k_new_seqlens.empty() && !cfg.cu_seqlens_k_new.empty()) {
    throw std::runtime_error("--k-new-seqlens and --cu-seqlens-k-new are mutually exclusive");
  }
  if (!cfg.k_new_seqlens.empty() &&
      !(static_cast<int>(cfg.k_new_seqlens.size()) == 1 || static_cast<int>(cfg.k_new_seqlens.size()) == cfg.batch)) {
    throw std::runtime_error("--k-new-seqlens must have one entry or exactly batch entries");
  }
  if (!cfg.cu_seqlens_k_new.empty() && static_cast<int>(cfg.cu_seqlens_k_new.size()) != cfg.batch + 1) {
    throw std::runtime_error("--cu-seqlens-k-new must have exactly batch+1 entries");
  }
  if (!cfg.cache_seqlens_old.empty() &&
      !(static_cast<int>(cfg.cache_seqlens_old.size()) == 1 ||
        static_cast<int>(cfg.cache_seqlens_old.size()) == cfg.batch)) {
    throw std::runtime_error("--cache-seqlens-old must have one entry or exactly batch entries");
  }
  if (!cfg.cache_seqlens_old.empty() && (cfg.past_kv_set || !cfg.past_kv_list.empty())) {
    throw std::runtime_error("--cache-seqlens-old is mutually exclusive with --past-kv/--past-kv-list");
  }
  for (int v : cfg.past_kv_list) {
    if (v < 0) throw std::runtime_error("--past-kv-list entries must be non-negative");
  }
  for (int v : cfg.seqlen_q_list) {
    if (v <= 0) throw std::runtime_error("--seqlen-q-list entries must be positive");
  }
  for (int v : cfg.k_new_seqlens) {
    if (v < 0) throw std::runtime_error("--k-new-seqlens entries must be non-negative");
  }
  for (int v : cfg.cache_seqlens_old) {
    if (v < 0) throw std::runtime_error("--cache-seqlens-old entries must be non-negative");
  }
  if (!cfg.cu_seqlens_k_new.empty()) {
    if (cfg.cu_seqlens_k_new.front() != 0) {
      throw std::runtime_error("--cu-seqlens-k-new must start with 0");
    }
    for (int i = 0; i < cfg.batch; ++i) {
      if (cfg.cu_seqlens_k_new[i + 1] < cfg.cu_seqlens_k_new[i]) {
        throw std::runtime_error("--cu-seqlens-k-new must be nondecreasing");
      }
    }
  }
  if (cfg.heads_q <= 0 || cfg.heads_kv <= 0 || cfg.heads_q % cfg.heads_kv != 0) {
    throw std::runtime_error("heads_q must be positive and divisible by heads_kv");
  }
  if (cfg.head_dim <= 0 || cfg.head_dim_v <= 0 || cfg.head_dim % 8 != 0 || cfg.head_dim_v % 8 != 0) {
    throw std::runtime_error("head_dim and head_dim_v must be positive multiples of 8");
  }
  if (cfg.paged && (cfg.page_size <= 0 || cfg.page_size % 64 != 0)) {
    throw std::runtime_error("paged prefill requires page_size to be a positive multiple of 64");
  }
  if (cfg.page_table_random && !cfg.paged) {
    throw std::runtime_error("--page-table-random requires --paged 1");
  }
  if (cfg.sink && (!cfg.paged || cfg.head_dim != 64)) {
    throw std::runtime_error("the current SGL prefill runner supports sink only on paged head_dim=64");
  }
  if (!cfg.paged && !(cfg.head_dim == 64 || cfg.head_dim == 72 || cfg.head_dim == 96 || cfg.head_dim == 128)) {
    throw std::runtime_error("non-paged standalone prefill supports head_dim in {64,72,96,128}");
  }
  if (cfg.paged &&
      !(cfg.head_dim == 64 || cfg.head_dim == 96 || cfg.head_dim == 128 || cfg.head_dim == 192 ||
        cfg.head_dim == 256 || cfg.head_dim == 512)) {
    throw std::runtime_error("paged standalone prefill supports head_dim in {64,96,128,192,256,512}");
  }
  return cfg;
}

LengthInfo make_lengths(const Config& cfg) {
  LengthInfo lengths;
  lengths.q_lens.resize(cfg.batch);
  lengths.k_lens.resize(cfg.batch);
  lengths.past_lens.resize(cfg.batch);
  lengths.k_new_lens.assign(cfg.batch, 0);
  lengths.cache_lens_old.assign(cfg.batch, 0);
  lengths.cu_q.assign(cfg.batch + 1, 0);
  lengths.cu_k.assign(cfg.batch + 1, 0);
  lengths.cu_k_new.assign(cfg.batch + 1, 0);
  lengths.append_kv = append_kv_enabled(cfg);
  lengths.k_new_uses_cu = !cfg.cu_seqlens_k_new.empty() || static_cast<int>(cfg.k_new_seqlens.size()) == cfg.batch;

  if (lengths.append_kv) {
    if (!cfg.cu_seqlens_k_new.empty()) {
      for (int b = 0; b < cfg.batch; ++b) {
        lengths.k_new_lens[b] = cfg.cu_seqlens_k_new[b + 1] - cfg.cu_seqlens_k_new[b];
        lengths.cu_k_new[b + 1] = cfg.cu_seqlens_k_new[b + 1];
      }
    } else if (cfg.k_new_seqlens.size() == 1) {
      for (int b = 0; b < cfg.batch; ++b) {
        lengths.k_new_lens[b] = cfg.k_new_seqlens[0];
        lengths.cu_k_new[b + 1] = lengths.cu_k_new[b] + lengths.k_new_lens[b];
      }
    } else {
      for (int b = 0; b < cfg.batch; ++b) {
        lengths.k_new_lens[b] = cfg.k_new_seqlens[b];
        lengths.cu_k_new[b + 1] = lengths.cu_k_new[b] + lengths.k_new_lens[b];
      }
    }
  }

  const bool chunk_lengths = cfg.past_kv_set || !cfg.past_kv_list.empty();
  for (int b = 0; b < cfg.batch; ++b) {
    lengths.q_lens[b] = cfg.seqlen_q_list.empty() ? cfg.seqlen_q : cfg.seqlen_q_list[b];
    if (lengths.append_kv) {
      int old_len = 0;
      if (!cfg.cache_seqlens_old.empty()) {
        old_len = cfg.cache_seqlens_old.size() == 1 ? cfg.cache_seqlens_old[0] : cfg.cache_seqlens_old[b];
      } else if (chunk_lengths) {
        old_len = cfg.past_kv_list.empty() ? cfg.past_kv : cfg.past_kv_list[b];
      } else {
        old_len = cfg.seqlen_k - lengths.k_new_lens[b];
      }
      if (old_len < 0) {
        throw std::runtime_error("resolved append cache_seqlens_old must be non-negative");
      }
      lengths.cache_lens_old[b] = old_len;
      lengths.past_lens[b] = old_len;
      lengths.k_lens[b] = old_len + lengths.k_new_lens[b];
    } else if (chunk_lengths) {
      lengths.past_lens[b] = cfg.past_kv_list.empty() ? cfg.past_kv : cfg.past_kv_list[b];
      lengths.k_lens[b] = lengths.past_lens[b] + lengths.q_lens[b];
    } else {
      lengths.k_lens[b] = cfg.seqlen_k;
      lengths.past_lens[b] = lengths.k_lens[b] - lengths.q_lens[b];
      lengths.cache_lens_old[b] = lengths.past_lens[b];
    }
    if (lengths.k_lens[b] <= 0) {
      throw std::runtime_error("resolved per-batch seqlen_k must be positive");
    }
    lengths.max_q = std::max(lengths.max_q, lengths.q_lens[b]);
    lengths.max_k = std::max(lengths.max_k, lengths.k_lens[b]);
    lengths.max_k_new = std::max(lengths.max_k_new, lengths.k_new_lens[b]);
    lengths.cu_q[b + 1] = lengths.cu_q[b] + lengths.q_lens[b];
    lengths.cu_k[b + 1] = lengths.cu_k[b] + lengths.k_lens[b];
  }
  lengths.total_q = lengths.cu_q.back();
  lengths.total_k_new = lengths.cu_k_new.back();
  if (lengths.append_kv && lengths.total_k_new <= 0) {
    throw std::runtime_error("append KV requires at least one k_new/v_new token");
  }

  if (cfg.paged) {
    std::vector<int32_t> pages_per_batch(cfg.batch);
    int max_pages = 0;
    for (int b = 0; b < cfg.batch; ++b) {
      pages_per_batch[b] = (lengths.k_lens[b] + cfg.page_size - 1) / cfg.page_size;
      max_pages = std::max(max_pages, pages_per_batch[b]);
      lengths.total_pages += pages_per_batch[b];
    }
    std::vector<int32_t> physical_pages(lengths.total_pages);
    std::iota(physical_pages.begin(), physical_pages.end(), 0);
    if (cfg.page_table_random) {
      std::mt19937 page_rng(static_cast<uint32_t>(cfg.seed) + 42u);
      std::shuffle(physical_pages.begin(), physical_pages.end(), page_rng);
    }
    // The mainloop uses max_num_pages_per_seq as the page-table row stride.
    // Keep the stride equal to the real number of logical pages per row so the
    // standalone ABI matches the FA3/FA4-style paged cache contract.
    lengths.page_table_stride = max_pages;
    lengths.page_table.assign(static_cast<std::size_t>(cfg.batch) * lengths.page_table_stride, 0);
    int logical_page = 0;
    for (int b = 0; b < cfg.batch; ++b) {
      for (int p = 0; p < pages_per_batch[b]; ++p) {
        lengths.page_table[static_cast<std::size_t>(b) * lengths.page_table_stride + p] =
            physical_pages[logical_page++];
      }
    }
    lengths.total_k = lengths.total_pages * cfg.page_size;
  } else {
    lengths.total_k = lengths.cu_k.back();
  }

  return lengths;
}

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  DeviceBuffer(sycl::queue& q, std::size_t n) : q_(&q), n_(n) {
    ptr_ = sycl::malloc_device<T>(n_, q);
    if (ptr_ == nullptr) {
      throw std::runtime_error("failed to allocate device buffer");
    }
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept { move_from(other); }
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      reset();
      move_from(other);
    }
    return *this;
  }
  ~DeviceBuffer() { reset(); }

  T* data() const { return ptr_; }
  std::size_t size() const { return n_; }

  void copy_from_host(const std::vector<T>& host) {
    if (host.size() != n_) throw std::runtime_error("host/device size mismatch");
    q_->memcpy(ptr_, host.data(), n_ * sizeof(T)).wait();
  }

  std::vector<T> copy_to_host() const {
    std::vector<T> host(n_);
    q_->memcpy(host.data(), ptr_, n_ * sizeof(T)).wait();
    return host;
  }

 private:
  void reset() {
    if (ptr_ != nullptr) {
      q_->wait();
      sycl::free(ptr_, *q_);
    }
    ptr_ = nullptr;
    n_ = 0;
    q_ = nullptr;
  }

  void move_from(DeviceBuffer& other) {
    q_ = other.q_;
    ptr_ = other.ptr_;
    n_ = other.n_;
    other.q_ = nullptr;
    other.ptr_ = nullptr;
    other.n_ = 0;
  }

  sycl::queue* q_ = nullptr;
  T* ptr_ = nullptr;
  std::size_t n_ = 0;
};

std::vector<bf16_t> make_random_bf16(std::size_t n, std::mt19937& rng) {
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<bf16_t> host(n);
  for (auto& x : host) {
    x = float_to_bf16(dist(rng));
  }
  return host;
}

struct LogicalKV {
  std::vector<bf16_t> k;
  std::vector<bf16_t> v;
};

LogicalKV make_reference_kv(
    const Config& cfg,
    const LengthInfo& lengths,
    const std::vector<bf16_t>& k_cache,
    const std::vector<bf16_t>& v_cache,
    const std::vector<bf16_t>& k_new,
    const std::vector<bf16_t>& v_new) {
  LogicalKV logical;
  logical.k.resize(static_cast<std::size_t>(lengths.cu_k.back()) * cfg.heads_kv * cfg.head_dim);
  logical.v.resize(static_cast<std::size_t>(lengths.cu_k.back()) * cfg.heads_kv * cfg.head_dim_v);

  auto cache_k_index = [&](int b, int ck, int h, int d) -> std::size_t {
    if (cfg.paged) {
      const int page = lengths.page_table[static_cast<std::size_t>(b) * lengths.page_table_stride + ck / cfg.page_size];
      const int offset = ck % cfg.page_size;
      return ((static_cast<std::size_t>(page) * cfg.page_size + offset) * cfg.heads_kv + h) * cfg.head_dim + d;
    }
    return (static_cast<std::size_t>(lengths.cu_k[b] + ck) * cfg.heads_kv + h) * cfg.head_dim + d;
  };
  auto cache_v_index = [&](int b, int ck, int h, int d) -> std::size_t {
    if (cfg.paged) {
      const int page = lengths.page_table[static_cast<std::size_t>(b) * lengths.page_table_stride + ck / cfg.page_size];
      const int offset = ck % cfg.page_size;
      return ((static_cast<std::size_t>(page) * cfg.page_size + offset) * cfg.heads_kv + h) * cfg.head_dim_v + d;
    }
    return (static_cast<std::size_t>(lengths.cu_k[b] + ck) * cfg.heads_kv + h) * cfg.head_dim_v + d;
  };

  for (int b = 0; b < cfg.batch; ++b) {
    const int old_len = lengths.cache_lens_old[b];
    const int new_len = lengths.k_new_lens[b];
    for (int ck = 0; ck < lengths.k_lens[b]; ++ck) {
      const bool is_appended = lengths.append_kv && ck >= old_len && ck < old_len + new_len;
      const int new_tok = ck - old_len;
      const int new_abs_tok = lengths.cu_k_new[b] + new_tok;
      for (int h = 0; h < cfg.heads_kv; ++h) {
        for (int d = 0; d < cfg.head_dim; ++d) {
          const std::size_t dst =
              (static_cast<std::size_t>(lengths.cu_k[b] + ck) * cfg.heads_kv + h) * cfg.head_dim + d;
          if (is_appended) {
            const std::size_t src =
                (static_cast<std::size_t>(new_abs_tok) * cfg.heads_kv + h) * cfg.head_dim + d;
            logical.k[dst] = k_new[src];
          } else {
            logical.k[dst] = k_cache[cache_k_index(b, ck, h, d)];
          }
        }
        for (int d = 0; d < cfg.head_dim_v; ++d) {
          const std::size_t dst =
              (static_cast<std::size_t>(lengths.cu_k[b] + ck) * cfg.heads_kv + h) * cfg.head_dim_v + d;
          if (is_appended) {
            const std::size_t src =
                (static_cast<std::size_t>(new_abs_tok) * cfg.heads_kv + h) * cfg.head_dim_v + d;
            logical.v[dst] = v_new[src];
          } else {
            logical.v[dst] = v_cache[cache_v_index(b, ck, h, d)];
          }
        }
      }
    }
  }

  return logical;
}

void dispatch_prefill(const prefill::Arguments& params) {
  const bool paged = params.page_table != nullptr;
  switch (params.d) {
#ifdef FMHA_STANDALONE_HAS_HD_64
    case 64:
      if (paged) {
#ifdef FMHA_STANDALONE_HAS_PAGED_HD_64
        DISPATCH_PREFILL_KERNEL(64);
        return;
#endif
      } else {
#ifdef FMHA_STANDALONE_HAS_NP_HD_64
        DISPATCH_PREFILL_NOPAGE_KERNEL(64);
        return;
#endif
      }
      break;
#endif
#ifdef FMHA_STANDALONE_HAS_HD_72
    case 72:
      if (paged) {
#ifdef FMHA_STANDALONE_HAS_PAGED_HD_72
        DISPATCH_PREFILL_KERNEL(72);
        return;
#endif
      } else {
#ifdef FMHA_STANDALONE_HAS_NP_HD_72
        DISPATCH_PREFILL_NOPAGE_KERNEL(72);
        return;
#endif
      }
      break;
#endif
#ifdef FMHA_STANDALONE_HAS_HD_80
    case 80:
      if (paged) {
#ifdef FMHA_STANDALONE_HAS_PAGED_HD_80
        DISPATCH_PREFILL_KERNEL(80);
        return;
#endif
      } else {
#ifdef FMHA_STANDALONE_HAS_NP_HD_80
        DISPATCH_PREFILL_NOPAGE_KERNEL(80);
        return;
#endif
      }
      break;
#endif
#ifdef FMHA_STANDALONE_HAS_HD_96
    case 96:
      if (paged) {
#ifdef FMHA_STANDALONE_HAS_PAGED_HD_96
        DISPATCH_PREFILL_KERNEL(96);
        return;
#endif
      } else {
#ifdef FMHA_STANDALONE_HAS_NP_HD_96
        DISPATCH_PREFILL_NOPAGE_KERNEL(96);
        return;
#endif
      }
      break;
#endif
#ifdef FMHA_STANDALONE_HAS_HD_128
    case 128:
      if (paged) {
#ifdef FMHA_STANDALONE_HAS_PAGED_HD_128
        DISPATCH_PREFILL_KERNEL(128);
        return;
#endif
      } else {
#ifdef FMHA_STANDALONE_HAS_NP_HD_128
        DISPATCH_PREFILL_NOPAGE_KERNEL(128);
        return;
#endif
      }
      break;
#endif
#ifdef FMHA_STANDALONE_HAS_HD_192
    case 192:
      if (paged) {
#ifdef FMHA_STANDALONE_HAS_PAGED_HD_192
        DISPATCH_PREFILL_KERNEL(192);
        return;
#endif
      } else {
#ifdef FMHA_STANDALONE_HAS_NP_HD_192
        DISPATCH_PREFILL_NOPAGE_KERNEL(192);
        return;
#endif
      }
      break;
#endif
#ifdef FMHA_STANDALONE_HAS_HD_256
    case 256:
      if (paged) {
#ifdef FMHA_STANDALONE_HAS_PAGED_HD_256
        DISPATCH_PREFILL_KERNEL(256);
        return;
#endif
      } else {
#ifdef FMHA_STANDALONE_HAS_NP_HD_256
        DISPATCH_PREFILL_NOPAGE_KERNEL(256);
        return;
#endif
      }
      break;
#endif
#ifdef FMHA_STANDALONE_HAS_HD_512
    case 512:
      if (paged) {
#ifdef FMHA_STANDALONE_HAS_PAGED_HD_512
        DISPATCH_PREFILL_KERNEL(512);
        return;
#endif
      } else {
#ifdef FMHA_STANDALONE_HAS_NP_HD_512
        DISPATCH_PREFILL_NOPAGE_KERNEL(512);
        return;
#endif
      }
      break;
#endif
    default:
      break;
  }
  std::ostringstream oss;
  oss << "unsupported head_dim=" << params.d << " for " << (paged ? "paged" : "non-paged") << " prefill";
  throw std::runtime_error(oss.str());
}

void run_prefill(
    const Config& cfg,
    const LengthInfo& lengths,
    bf16_t* q,
    bf16_t* k,
    bf16_t* v,
    bf16_t* k_new,
    bf16_t* v_new,
    bf16_t* out,
    int32_t* cu_q,
    int32_t* cu_k_or_cache_lens,
    int32_t* cu_k_new,
    int32_t* cache_seqlens_old,
    int32_t* page_table,
    bf16_t* sinks) {
  int window_left = cfg.window_left;
  int window_right = cfg.window_right;
  if (window_left >= lengths.max_k - 1) window_left = -1;
  window_right = std::min(window_right, lengths.max_q);
  if (cfg.causal) window_right = 0;

  prefill::Arguments params{};
  params.is_bf16 = true;
  params.q_ptr = q;
  params.k_ptr = k;
  params.v_ptr = v;
  params.o_ptr = out;
  params.softmax_sink_ptr = cfg.sink ? sinks : nullptr;
  params.skip_batch_mask_ptr = nullptr;
  params.cu_seqlens_q = reinterpret_cast<int*>(cu_q);
  params.cu_seqlens_k = reinterpret_cast<int*>(cu_k_or_cache_lens);
  params.cu_seqlens_knew = lengths.k_new_uses_cu ? reinterpret_cast<int*>(cu_k_new) : nullptr;
  params.cache_seqlens_old = reinterpret_cast<int*>(cache_seqlens_old);
  params.b = cfg.batch;
  params.h = cfg.heads_q;
  params.h_k = cfg.heads_kv;
  params.q_group_size = 1;
  params.seqlen_q = lengths.max_q;
  params.seqlen_k = lengths.max_k;
  params.seqlen_knew = lengths.max_k_new;
  params.total_q = lengths.total_q;
  params.total_k = lengths.total_k;
  params.total_knew = lengths.total_k_new;
  params.b_k = cfg.batch;
  params.d = cfg.head_dim;
  params.d_rounded = round_up_headdim_standalone(cfg.head_dim);
  params.dv = cfg.head_dim_v;
  params.dv_rounded = round_up_headdim_standalone(cfg.head_dim_v);
  params.softmax_scale = 1.0f / std::sqrt(static_cast<float>(cfg.head_dim));
  params.softcap = 0.0f;
  params.p_dropout = 1.0f;
  params.is_causal = window_left < 0 && window_right == 0;
  params.is_local = (window_left >= 0 || window_right >= 0) && !params.is_causal;
  if (window_left < 0) window_left = lengths.max_k - 1;
  if (window_right < 0) window_right = lengths.max_q - 1;
  params.window_size_left = window_left;
  params.window_size_right = window_right;
  params.page_table = cfg.paged ? reinterpret_cast<int*>(page_table) : nullptr;
  params.page_table_batch_stride = cfg.paged ? lengths.page_table_stride : 0;
  params.max_num_pages_per_seq = cfg.paged ? lengths.page_table_stride : 0;
  params.page_size = cfg.paged ? cfg.page_size : 0;
  params.num_pages = cfg.paged ? lengths.total_pages : 0;
  params.page_table_contiguous = cfg.paged && !cfg.page_table_random;
  params.page_table_identity =
      cfg.paged && !cfg.page_table_random && cfg.batch == 1 && !lengths.page_table.empty() && lengths.page_table[0] == 0;
  params.rotary_dim = 0;
  params.knew_ptr = k_new;
  params.vnew_ptr = v_new;

  dispatch_prefill(params);
}

std::vector<float> reference_prefill(
    const Config& cfg,
    const LengthInfo& lengths,
    const std::vector<bf16_t>& q,
    const std::vector<bf16_t>& k_logical,
    const std::vector<bf16_t>& v_logical,
    const std::vector<bf16_t>& sinks) {
  std::vector<float> ref(static_cast<std::size_t>(lengths.total_q) * cfg.heads_q * cfg.head_dim_v, 0.0f);
  const int head_group = cfg.heads_q / cfg.heads_kv;
  const float scale = 1.0f / std::sqrt(static_cast<float>(cfg.head_dim));

  auto q_at = [&](int row, int h, int d) -> float {
    return bf16_to_float(q[(static_cast<std::size_t>(row) * cfg.heads_q + h) * cfg.head_dim + d]);
  };
  auto k_at = [&](int b, int ck, int h, int d) -> float {
    return bf16_to_float(
        k_logical[(static_cast<std::size_t>(lengths.cu_k[b] + ck) * cfg.heads_kv + h) * cfg.head_dim + d]);
  };
  auto v_at = [&](int b, int ck, int h, int d) -> float {
    return bf16_to_float(
        v_logical[(static_cast<std::size_t>(lengths.cu_k[b] + ck) * cfg.heads_kv + h) * cfg.head_dim_v + d]);
  };

  for (int b = 0; b < cfg.batch; ++b) {
    const int q_len = lengths.q_lens[b];
    const int k_len = lengths.k_lens[b];
    for (int rq = 0; rq < q_len; ++rq) {
      const int q_row = lengths.cu_q[b] + rq;
      const int row_kv = lengths.append_kv ? (lengths.k_lens[b] - q_len + rq) : (lengths.past_lens[b] + rq);
      for (int hq = 0; hq < cfg.heads_q; ++hq) {
        const int hk = hq / head_group;
        std::vector<float> scores(k_len, -std::numeric_limits<float>::infinity());
        float max_score = -std::numeric_limits<float>::infinity();
        for (int ck = 0; ck < k_len; ++ck) {
          bool keep = true;
          if (cfg.causal && ck > row_kv) keep = false;
          if (cfg.window_left >= 0 && ck < row_kv - cfg.window_left) keep = false;
          if (cfg.window_right >= 0 && ck > row_kv + cfg.window_right) keep = false;
          if (!keep) continue;

          float dot = 0.0f;
          for (int d = 0; d < cfg.head_dim; ++d) {
            dot += q_at(q_row, hq, d) * k_at(b, ck, hk, d);
          }
          scores[ck] = dot * scale;
          max_score = std::max(max_score, scores[ck]);
        }

        if (cfg.sink) {
          max_score = std::max(max_score, bf16_to_float(sinks[hq]));
        }
        if (!std::isfinite(max_score)) continue;

        float denom = 0.0f;
        if (cfg.sink) {
          denom += std::exp(bf16_to_float(sinks[hq]) - max_score);
        }
        std::vector<float> probs(k_len, 0.0f);
        for (int ck = 0; ck < k_len; ++ck) {
          if (std::isfinite(scores[ck])) {
            probs[ck] = std::exp(scores[ck] - max_score);
            denom += probs[ck];
          }
        }
        if (denom == 0.0f) continue;

        for (int dv = 0; dv < cfg.head_dim_v; ++dv) {
          float acc = 0.0f;
          for (int ck = 0; ck < k_len; ++ck) {
            if (probs[ck] != 0.0f) {
              acc += (probs[ck] / denom) * v_at(b, ck, hk, dv);
            }
          }
          ref[(static_cast<std::size_t>(q_row) * cfg.heads_q + hq) * cfg.head_dim_v + dv] = acc;
        }
      }
    }
  }
  return ref;
}

bool verify_output(const Config& cfg, const std::vector<bf16_t>& out, const std::vector<float>& ref) {
  double max_abs = 0.0;
  double max_ref = 0.0;
  int64_t bad_count = 0;
  for (std::size_t i = 0; i < out.size(); ++i) {
    const double actual = bf16_to_float(out[i]);
    const double expected = ref[i];
    const double diff = std::abs(actual - expected);
    const double tol = std::abs(expected) * cfg.rtol + cfg.atol;
    max_abs = std::max(max_abs, diff);
    max_ref = std::max(max_ref, std::abs(expected));
    if (diff > tol) ++bad_count;
  }
  const double max_rel = max_abs / std::max(max_ref, 1e-12);
  std::cout << "verify: max_abs=" << max_abs << " max_rel=" << max_rel << " bad=" << bad_count << "/"
            << out.size() << " atol=" << cfg.atol << " rtol=" << cfg.rtol << "\n";
  return bad_count == 0;
}

bool verify_append_cache(
    const Config& cfg,
    const LengthInfo& lengths,
    const std::vector<bf16_t>& k_cache,
    const std::vector<bf16_t>& v_cache,
    const std::vector<bf16_t>& k_new,
    const std::vector<bf16_t>& v_new) {
  if (!lengths.append_kv) {
    return true;
  }

  int64_t k_bad = 0;
  int64_t v_bad = 0;
  int64_t k_total = 0;
  int64_t v_total = 0;
  auto k_cache_index = [&](int b, int tok, int h, int d) -> std::size_t {
    if (cfg.paged) {
      const int page = lengths.page_table[static_cast<std::size_t>(b) * lengths.page_table_stride + tok / cfg.page_size];
      const int offset = tok % cfg.page_size;
      return ((static_cast<std::size_t>(page) * cfg.page_size + offset) * cfg.heads_kv + h) * cfg.head_dim + d;
    }
    return (static_cast<std::size_t>(lengths.cu_k[b] + tok) * cfg.heads_kv + h) * cfg.head_dim + d;
  };
  auto v_cache_index = [&](int b, int tok, int h, int d) -> std::size_t {
    if (cfg.paged) {
      const int page = lengths.page_table[static_cast<std::size_t>(b) * lengths.page_table_stride + tok / cfg.page_size];
      const int offset = tok % cfg.page_size;
      return ((static_cast<std::size_t>(page) * cfg.page_size + offset) * cfg.heads_kv + h) * cfg.head_dim_v + d;
    }
    return (static_cast<std::size_t>(lengths.cu_k[b] + tok) * cfg.heads_kv + h) * cfg.head_dim_v + d;
  };

  for (int b = 0; b < cfg.batch; ++b) {
    for (int nt = 0; nt < lengths.k_new_lens[b]; ++nt) {
      const int dst_tok = lengths.cache_lens_old[b] + nt;
      const int new_abs_tok = lengths.cu_k_new[b] + nt;
      for (int h = 0; h < cfg.heads_kv; ++h) {
        for (int d = 0; d < cfg.head_dim; ++d) {
          const std::size_t actual = k_cache_index(b, dst_tok, h, d);
          const std::size_t expected =
              (static_cast<std::size_t>(new_abs_tok) * cfg.heads_kv + h) * cfg.head_dim + d;
          if (k_cache[actual] != k_new[expected]) ++k_bad;
          ++k_total;
        }
        for (int d = 0; d < cfg.head_dim_v; ++d) {
          const std::size_t actual = v_cache_index(b, dst_tok, h, d);
          const std::size_t expected =
              (static_cast<std::size_t>(new_abs_tok) * cfg.heads_kv + h) * cfg.head_dim_v + d;
          if (v_cache[actual] != v_new[expected]) ++v_bad;
          ++v_total;
        }
      }
    }
  }

  std::cout << "append_verify: k_bad=" << k_bad << "/" << k_total << " v_bad=" << v_bad << "/" << v_total
            << "\n";
  return k_bad == 0 && v_bad == 0;
}

double estimate_tflops(const Config& cfg, const LengthInfo& lengths, double ms) {
  int64_t qk_tokens = 0;
  for (int b = 0; b < cfg.batch; ++b) {
    qk_tokens += static_cast<int64_t>(lengths.q_lens[b]) * lengths.k_lens[b];
  }
  const double flops_qk = 2.0 * cfg.heads_q * qk_tokens * cfg.head_dim;
  const double flops_pv = 2.0 * cfg.heads_q * qk_tokens * cfg.head_dim_v;
  return (flops_qk + flops_pv) / (ms * 1.0e9);
}

double event_duration_ms(const sycl::event& event) {
  const auto start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
  const auto end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
  return static_cast<double>(end - start) * 1.0e-6;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Config cfg = parse_args(argc, argv);
    sycl::queue q{
        sycl::gpu_selector_v,
        [](sycl::exception_list exceptions) {
          for (const auto& e : exceptions) {
            try {
              std::rethrow_exception(e);
            } catch (const sycl::exception& ex) {
              std::cerr << "asynchronous SYCL exception: " << ex.what() << "\n";
            }
          }
        },
        sycl::property_list{sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}}};
    sgl_standalone::set_queue(&q);

    std::mt19937 rng(cfg.seed);
    const LengthInfo lengths = make_lengths(cfg);

    std::vector<bf16_t> q_host =
        make_random_bf16(static_cast<std::size_t>(lengths.total_q) * cfg.heads_q * cfg.head_dim, rng);
    std::vector<bf16_t> k_host;
    std::vector<bf16_t> v_host;
    std::vector<bf16_t> k_new_host;
    std::vector<bf16_t> v_new_host;
    std::vector<int32_t> cu_k_or_cache_lens_host;
    if (cfg.paged) {
      k_host = make_random_bf16(static_cast<std::size_t>(lengths.total_k) * cfg.heads_kv * cfg.head_dim, rng);
      v_host = make_random_bf16(static_cast<std::size_t>(lengths.total_k) * cfg.heads_kv * cfg.head_dim_v, rng);
      cu_k_or_cache_lens_host = lengths.k_lens;
    } else {
      k_host = make_random_bf16(static_cast<std::size_t>(lengths.total_k) * cfg.heads_kv * cfg.head_dim, rng);
      v_host = make_random_bf16(static_cast<std::size_t>(lengths.total_k) * cfg.heads_kv * cfg.head_dim_v, rng);
      cu_k_or_cache_lens_host = lengths.cu_k;
    }
    if (lengths.append_kv) {
      k_new_host =
          make_random_bf16(static_cast<std::size_t>(lengths.total_k_new) * cfg.heads_kv * cfg.head_dim, rng);
      v_new_host =
          make_random_bf16(static_cast<std::size_t>(lengths.total_k_new) * cfg.heads_kv * cfg.head_dim_v, rng);
    }
    std::vector<bf16_t> sinks_host = cfg.sink ? make_random_bf16(cfg.heads_q, rng) : std::vector<bf16_t>{};

    DeviceBuffer<bf16_t> q_dev(q, q_host.size());
    DeviceBuffer<bf16_t> k_dev(q, k_host.size());
    DeviceBuffer<bf16_t> v_dev(q, v_host.size());
    DeviceBuffer<bf16_t> k_new_dev;
    DeviceBuffer<bf16_t> v_new_dev;
    DeviceBuffer<bf16_t> out_dev(q, static_cast<std::size_t>(lengths.total_q) * cfg.heads_q * cfg.head_dim_v);
    DeviceBuffer<int32_t> cu_q_dev(q, lengths.cu_q.size());
    DeviceBuffer<int32_t> cu_k_dev(q, cu_k_or_cache_lens_host.size());
    DeviceBuffer<int32_t> cu_k_new_dev;
    DeviceBuffer<int32_t> cache_seqlens_old_dev;
    DeviceBuffer<int32_t> page_table_dev;
    DeviceBuffer<bf16_t> sinks_dev;

    q_dev.copy_from_host(q_host);
    k_dev.copy_from_host(k_host);
    v_dev.copy_from_host(v_host);
    cu_q_dev.copy_from_host(lengths.cu_q);
    cu_k_dev.copy_from_host(cu_k_or_cache_lens_host);
    if (lengths.append_kv) {
      k_new_dev = DeviceBuffer<bf16_t>(q, k_new_host.size());
      v_new_dev = DeviceBuffer<bf16_t>(q, v_new_host.size());
      cache_seqlens_old_dev = DeviceBuffer<int32_t>(q, lengths.cache_lens_old.size());
      k_new_dev.copy_from_host(k_new_host);
      v_new_dev.copy_from_host(v_new_host);
      cache_seqlens_old_dev.copy_from_host(lengths.cache_lens_old);
      if (lengths.k_new_uses_cu) {
        cu_k_new_dev = DeviceBuffer<int32_t>(q, lengths.cu_k_new.size());
        cu_k_new_dev.copy_from_host(lengths.cu_k_new);
      }
    }
    if (cfg.paged) {
      page_table_dev = DeviceBuffer<int32_t>(q, lengths.page_table.size());
      page_table_dev.copy_from_host(lengths.page_table);
    }
    if (cfg.sink) {
      sinks_dev = DeviceBuffer<bf16_t>(q, sinks_host.size());
      sinks_dev.copy_from_host(sinks_host);
    }

    std::cout << "device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";
    std::cout << "shape: batch=" << cfg.batch << " sq=" << lengths.max_q << " sk=" << lengths.max_k
              << " total_q=" << lengths.total_q << " total_k=" << lengths.total_k
              << " append=" << lengths.append_kv << " total_k_new=" << lengths.total_k_new
              << " hq=" << cfg.heads_q << " hkv=" << cfg.heads_kv << " d=" << cfg.head_dim
              << " dv=" << cfg.head_dim_v << " paged=" << cfg.paged << " page_size=" << cfg.page_size
              << " page_stride=" << lengths.page_table_stride << " page_table_random=" << cfg.page_table_random
              << " causal=" << cfg.causal << " window=(" << cfg.window_left << "," << cfg.window_right
              << ") sink=" << cfg.sink << "\n";

    auto launch_once = [&] {
      run_prefill(
          cfg,
          lengths,
          q_dev.data(),
          k_dev.data(),
          v_dev.data(),
          lengths.append_kv ? k_new_dev.data() : nullptr,
          lengths.append_kv ? v_new_dev.data() : nullptr,
          out_dev.data(),
          cu_q_dev.data(),
          cu_k_dev.data(),
          lengths.k_new_uses_cu ? cu_k_new_dev.data() : nullptr,
          lengths.append_kv ? cache_seqlens_old_dev.data() : nullptr,
          cfg.paged ? page_table_dev.data() : nullptr,
          cfg.sink ? sinks_dev.data() : nullptr);
      return sgl_standalone::last_event();
    };

    auto first_event = launch_once();
    first_event.wait();
    q.wait();

    if (cfg.verify) {
      LogicalKV logical_kv = make_reference_kv(cfg, lengths, k_host, v_host, k_new_host, v_new_host);
      std::vector<float> ref = reference_prefill(cfg, lengths, q_host, logical_kv.k, logical_kv.v, sinks_host);
      std::vector<bf16_t> out_host = out_dev.copy_to_host();
      if (!verify_output(cfg, out_host, ref)) {
        sgl_standalone::release_workspace();
        return 1;
      }
      if (lengths.append_kv) {
        std::vector<bf16_t> k_after = k_dev.copy_to_host();
        std::vector<bf16_t> v_after = v_dev.copy_to_host();
        if (!verify_append_cache(cfg, lengths, k_after, v_after, k_new_host, v_new_host)) {
          sgl_standalone::release_workspace();
          return 1;
        }
      }
    }

    for (int i = 0; i < cfg.warmup; ++i) {
      auto event = launch_once();
      event.wait();
    }

    // Measure device time over every kernel the prefill call enqueues: a single
    // launch may dispatch more than one kernel, in which case timing only the
    // last event would omit most of the work. clear_events() drops the warmup
    // events so only the measured iterations are summed.
    sgl_standalone::clear_events();
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < cfg.iters; ++i) {
      auto event = launch_once();
      event.wait();
    }
    const auto end = std::chrono::steady_clock::now();
    const double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double kernel_total_ms = 0.0;
    for (const auto& event : sgl_standalone::recorded_events()) {
      kernel_total_ms += event_duration_ms(event);
    }
    const double host_avg_ms = total_ms / std::max(cfg.iters, 1);
    const double kernel_avg_ms = kernel_total_ms / std::max(cfg.iters, 1);
    // Per-launch breakdown, for shapes that dispatch more than one kernel. The
    // events arrive grouped by iteration, so averaging every k-th event gives
    // the cost of the k-th kernel in the dispatch.
    if (std::getenv("FMHA_PROFILE_PER_LAUNCH") != nullptr) {
      const auto& events = sgl_standalone::recorded_events();
      const int per_iter = static_cast<int>(events.size()) / std::max(cfg.iters, 1);
      if (per_iter > 1) {
        std::cout << "per_launch:";
        for (int k = 0; k < per_iter; ++k) {
          double sum = 0.0;
          for (int i = 0; i < cfg.iters; ++i) {
            sum += event_duration_ms(events[i * per_iter + k]);
          }
          std::cout << " k" << k << "=" << sum / std::max(cfg.iters, 1);
        }
        std::cout << "\n";
      }
    }
    std::cout << "profile: kernel_avg_ms=" << kernel_avg_ms << " host_avg_ms=" << host_avg_ms
              << " iters=" << cfg.iters << " estimated_tflops=" << estimate_tflops(cfg, lengths, kernel_avg_ms)
              << "\n";

    sgl_standalone::release_workspace();
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 2;
  }
}
