#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
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
  int heads_q = 16;
  int heads_kv = 4;
  int head_dim = 64;
  int head_dim_v = 64;
  int page_size = 64;
  bool paged = true;
  bool causal = true;
  int window_left = -1;
  int window_right = -1;
  bool sink = false;
  bool relative_bias = false;
  int rel_extent = 1024;
  int warmup = 5;
  int iters = 5;
  int seed = 0;
  double atol = 5e-2;
  double rtol = 5e-2;
  bool verify = true;
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

Config parse_args(int argc, char** argv) {
  Config cfg;
  std::unordered_map<std::string, std::string> kv;
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg == "--help" || arg == "-h") {
      std::cout
          << "Usage: fmha_prefill_kvcache [options]\n"
          << "  --batch N --seqlen-q N --seqlen-k N --heads-q N --heads-kv N\n"
          << "  --head-dim N --head-dim-v N --paged 0|1 --page-size N\n"
          << "  --causal 0|1 --window-left N --window-right N --sink 0|1\n"
          << "  --relative-bias 0|1 --rel-extent N\n"
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

  get_int("batch", cfg.batch);
  get_int("seqlen-q", cfg.seqlen_q);
  get_int("seqlen-k", cfg.seqlen_k);
  get_int("heads-q", cfg.heads_q);
  get_int("heads-kv", cfg.heads_kv);
  get_int("head-dim", cfg.head_dim);
  get_int("head-dim-v", cfg.head_dim_v);
  get_int("page-size", cfg.page_size);
  get_bool("paged", cfg.paged);
  get_bool("causal", cfg.causal);
  get_int("window-left", cfg.window_left);
  get_int("window-right", cfg.window_right);
  get_bool("sink", cfg.sink);
  get_bool("relative-bias", cfg.relative_bias);
  get_int("rel-extent", cfg.rel_extent);
  get_int("warmup", cfg.warmup);
  get_int("iters", cfg.iters);
  get_int("seed", cfg.seed);
  get_double("atol", cfg.atol);
  get_double("rtol", cfg.rtol);
  get_bool("verify", cfg.verify);

  if (cfg.batch <= 0 || cfg.seqlen_q <= 0 || cfg.seqlen_k <= 0) {
    throw std::runtime_error("batch, seqlen_q, and seqlen_k must be positive");
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
  if (cfg.sink && (!cfg.paged || cfg.head_dim != 64)) {
    throw std::runtime_error("the current SGL prefill runner supports sink only on paged head_dim=64");
  }
  if (cfg.relative_bias &&
      (!cfg.paged || cfg.head_dim != 128 || cfg.head_dim_v != 128 || cfg.rel_extent <= 0)) {
    throw std::runtime_error(
        "relative attention requires paged=1, head_dim=head_dim_v=128, and rel_extent>0");
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

std::vector<int32_t> make_prefix_lengths(int batch, int seqlen) {
  std::vector<int32_t> host(batch + 1);
  for (int i = 0; i <= batch; ++i) host[i] = i * seqlen;
  return host;
}

std::vector<int32_t> make_cache_lengths(int batch, int seqlen) {
  return std::vector<int32_t>(batch, seqlen);
}

std::vector<int32_t> make_identity_page_table(int batch, int pages_per_seq) {
  std::vector<int32_t> table(batch * pages_per_seq);
  for (int b = 0; b < batch; ++b) {
    for (int p = 0; p < pages_per_seq; ++p) {
      table[b * pages_per_seq + p] = b * pages_per_seq + p;
    }
  }
  return table;
}

std::vector<bf16_t> make_random_bf16(std::size_t n, std::mt19937& rng) {
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<bf16_t> host(n);
  for (auto& x : host) {
    x = float_to_bf16(dist(rng));
  }
  return host;
}

void dispatch_prefill(const prefill::Arguments& params) {
  switch (params.d) {
#ifdef FMHA_STANDALONE_HAS_HD_64
    case 64:
      DISPATCH_PREFILL_KERNEL(64);
      break;
#endif
#ifdef FMHA_STANDALONE_HAS_HD_72
    case 72:
      DISPATCH_PREFILL_KERNEL(72);
      break;
#endif
#ifdef FMHA_STANDALONE_HAS_HD_96
    case 96:
      DISPATCH_PREFILL_KERNEL(96);
      break;
#endif
#ifdef FMHA_STANDALONE_HAS_HD_128
    case 128:
      DISPATCH_PREFILL_KERNEL(128);
      break;
#endif
#ifdef FMHA_STANDALONE_HAS_HD_192
    case 192:
      DISPATCH_PREFILL_KERNEL(192);
      break;
#endif
#ifdef FMHA_STANDALONE_HAS_HD_256
    case 256:
      DISPATCH_PREFILL_KERNEL(256);
      break;
#endif
#ifdef FMHA_STANDALONE_HAS_HD_512
    case 512:
      DISPATCH_PREFILL_KERNEL(512);
      break;
#endif
    default:
      throw std::runtime_error("unsupported head_dim");
  }
}

void run_prefill(
    const Config& cfg,
    bf16_t* q,
    bf16_t* k,
    bf16_t* v,
    bf16_t* out,
    int32_t* cu_q,
    int32_t* cu_k_or_cache_lens,
    int32_t* page_table,
    bf16_t* sinks,
    bf16_t* rel_bias) {
  const int total_q = cfg.batch * cfg.seqlen_q;
  const int pages_per_seq = cfg.paged ? (cfg.seqlen_k + cfg.page_size - 1) / cfg.page_size : 0;
  const int num_pages = cfg.paged ? cfg.batch * pages_per_seq : 0;
  const int seqlen_k_extent = cfg.paged ? pages_per_seq * cfg.page_size : cfg.seqlen_k;

  int window_left = cfg.window_left;
  int window_right = cfg.window_right;
  if (window_left >= seqlen_k_extent - 1) window_left = -1;
  window_right = std::min(window_right, cfg.seqlen_q);
  if (cfg.causal) window_right = 0;

  prefill::Arguments params{};
  params.is_bf16 = true;
  params.q_ptr = q;
  params.k_ptr = k;
  params.v_ptr = v;
  params.o_ptr = out;
  params.softmax_sink_ptr = cfg.sink ? sinks : nullptr;
  params.rel_bias_ptr = cfg.relative_bias ? rel_bias : nullptr;
  // Sheared [total_q, h, rel_bias_padded_cols(rel_extent)] bias.
  const int64_t rel_bias_cols = prefill::rel_bias_padded_cols(cfg.rel_extent);
  params.rel_bias_token_stride = static_cast<int64_t>(cfg.heads_q) * rel_bias_cols;
  params.rel_bias_head_stride = rel_bias_cols;
  params.rel_bias_extent = cfg.relative_bias ? cfg.rel_extent : 0;
  params.skip_batch_mask_ptr = nullptr;
  params.cu_seqlens_q = reinterpret_cast<int*>(cu_q);
  params.cu_seqlens_k = reinterpret_cast<int*>(cu_k_or_cache_lens);
  params.cu_seqlens_knew = nullptr;
  params.b = cfg.batch;
  params.h = cfg.heads_q;
  params.h_k = cfg.heads_kv;
  params.q_group_size = 1;
  params.seqlen_q = cfg.seqlen_q;
  params.seqlen_k = seqlen_k_extent;
  params.seqlen_knew = 0;
  params.total_q = total_q;
  params.total_k = cfg.paged ? num_pages * cfg.page_size : cfg.batch * cfg.seqlen_k;
  params.total_knew = 0;
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
  if (window_left < 0) window_left = seqlen_k_extent - 1;
  if (window_right < 0) window_right = cfg.seqlen_q - 1;
  params.window_size_left = window_left;
  params.window_size_right = window_right;
  params.page_table = cfg.paged ? reinterpret_cast<int*>(page_table) : nullptr;
  params.page_table_batch_stride = cfg.paged ? pages_per_seq : 0;
  params.max_num_pages_per_seq = pages_per_seq;
  params.page_size = cfg.paged ? cfg.page_size : 0;
  params.num_pages = num_pages;
  params.rotary_dim = 0;

  dispatch_prefill(params);
}

std::vector<float> reference_prefill(
    const Config& cfg,
    const std::vector<bf16_t>& q,
    const std::vector<bf16_t>& k,
    const std::vector<bf16_t>& v,
    const std::vector<int32_t>& page_table,
    const std::vector<bf16_t>& sinks,
    const std::vector<bf16_t>& rel_bias) {
  std::vector<float> ref(static_cast<std::size_t>(cfg.batch) * cfg.seqlen_q * cfg.heads_q * cfg.head_dim_v, 0.0f);
  const int head_group = cfg.heads_q / cfg.heads_kv;
  const float scale = 1.0f / std::sqrt(static_cast<float>(cfg.head_dim));
  const int pages_per_seq = cfg.paged ? (cfg.seqlen_k + cfg.page_size - 1) / cfg.page_size : 0;

  auto q_at = [&](int row, int h, int d) -> float {
    return bf16_to_float(q[(static_cast<std::size_t>(row) * cfg.heads_q + h) * cfg.head_dim + d]);
  };
  auto k_at = [&](int b, int ck, int h, int d) -> float {
    if (cfg.paged) {
      const int page = page_table[b * pages_per_seq + ck / cfg.page_size];
      const int offset = ck % cfg.page_size;
      return bf16_to_float(k[((static_cast<std::size_t>(page) * cfg.page_size + offset) * cfg.heads_kv + h) *
                             cfg.head_dim +
                             d]);
    }
    return bf16_to_float(k[(static_cast<std::size_t>(b * cfg.seqlen_k + ck) * cfg.heads_kv + h) * cfg.head_dim + d]);
  };
  auto v_at = [&](int b, int ck, int h, int d) -> float {
    if (cfg.paged) {
      const int page = page_table[b * pages_per_seq + ck / cfg.page_size];
      const int offset = ck % cfg.page_size;
      return bf16_to_float(v[((static_cast<std::size_t>(page) * cfg.page_size + offset) * cfg.heads_kv + h) *
                             cfg.head_dim_v +
                             d]);
    }
    return bf16_to_float(
        v[(static_cast<std::size_t>(b * cfg.seqlen_k + ck) * cfg.heads_kv + h) * cfg.head_dim_v + d]);
  };

  for (int b = 0; b < cfg.batch; ++b) {
    for (int rq = 0; rq < cfg.seqlen_q; ++rq) {
      const int q_row = b * cfg.seqlen_q + rq;
      const int row_kv = cfg.seqlen_k - cfg.seqlen_q + rq;
      for (int hq = 0; hq < cfg.heads_q; ++hq) {
        const int hk = hq / head_group;
        std::vector<float> scores(cfg.seqlen_k, -std::numeric_limits<float>::infinity());
        float max_score = -std::numeric_limits<float>::infinity();
        for (int ck = 0; ck < cfg.seqlen_k; ++ck) {
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
          if (cfg.relative_bias) {
            const int rel = row_kv - ck;
            if (rel >= 0 && rel < cfg.rel_extent) {
              const auto bias_idx =
                  (static_cast<std::size_t>(q_row) * cfg.heads_q + hq) * cfg.rel_extent + rel;
              scores[ck] += bf16_to_float(rel_bias[bias_idx]);
            }
          }
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
        std::vector<float> probs(cfg.seqlen_k, 0.0f);
        for (int ck = 0; ck < cfg.seqlen_k; ++ck) {
          if (std::isfinite(scores[ck])) {
            probs[ck] = std::exp(scores[ck] - max_score);
            denom += probs[ck];
          }
        }
        if (denom == 0.0f) continue;

        for (int dv = 0; dv < cfg.head_dim_v; ++dv) {
          float acc = 0.0f;
          for (int ck = 0; ck < cfg.seqlen_k; ++ck) {
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

double estimate_tflops(const Config& cfg, double ms) {
  const double flops_qk = 2.0 * cfg.batch * cfg.heads_q * cfg.seqlen_q * cfg.seqlen_k * cfg.head_dim;
  const double flops_pv = 2.0 * cfg.batch * cfg.heads_q * cfg.seqlen_q * cfg.seqlen_k * cfg.head_dim_v;
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
    const int total_q = cfg.batch * cfg.seqlen_q;
    const int pages_per_seq = cfg.paged ? (cfg.seqlen_k + cfg.page_size - 1) / cfg.page_size : 0;
    const int num_pages = cfg.paged ? cfg.batch * pages_per_seq : 0;

    std::vector<bf16_t> q_host = make_random_bf16(static_cast<std::size_t>(total_q) * cfg.heads_q * cfg.head_dim, rng);
    std::vector<bf16_t> k_host;
    std::vector<bf16_t> v_host;
    std::vector<int32_t> page_table_host;
    std::vector<int32_t> cu_k_or_cache_lens_host;
    if (cfg.paged) {
      k_host = make_random_bf16(static_cast<std::size_t>(num_pages) * cfg.page_size * cfg.heads_kv * cfg.head_dim, rng);
      v_host =
          make_random_bf16(static_cast<std::size_t>(num_pages) * cfg.page_size * cfg.heads_kv * cfg.head_dim_v, rng);
      page_table_host = make_identity_page_table(cfg.batch, pages_per_seq);
      cu_k_or_cache_lens_host = make_cache_lengths(cfg.batch, cfg.seqlen_k);
    } else {
      k_host =
          make_random_bf16(static_cast<std::size_t>(cfg.batch) * cfg.seqlen_k * cfg.heads_kv * cfg.head_dim, rng);
      v_host =
          make_random_bf16(static_cast<std::size_t>(cfg.batch) * cfg.seqlen_k * cfg.heads_kv * cfg.head_dim_v, rng);
      cu_k_or_cache_lens_host = make_prefix_lengths(cfg.batch, cfg.seqlen_k);
    }
    std::vector<int32_t> cu_q_host = make_prefix_lengths(cfg.batch, cfg.seqlen_q);
    std::vector<bf16_t> sinks_host = cfg.sink ? make_random_bf16(cfg.heads_q, rng) : std::vector<bf16_t>{};
    std::vector<bf16_t> rel_bias_host =
        cfg.relative_bias
            ? make_random_bf16(static_cast<std::size_t>(total_q) * cfg.heads_q * cfg.rel_extent, rng)
            : std::vector<bf16_t>{};
    std::vector<bf16_t> sheared_rel_bias_host;
    if (cfg.relative_bias) {
      // Stand-in for the shearing kernel: right-align each Q tile's band into a
      // k_tile-aligned column window, so the kernel reads a rectangle where the band is a
      // diagonal.  Out-of-band columns are zero; the masks drive those scores to -inf
      // independently, so the value only has to be finite.
      const int bias_cols = prefill::rel_bias_padded_cols(cfg.rel_extent);
      sheared_rel_bias_host.resize(
          static_cast<std::size_t>(total_q) * cfg.heads_q * bias_cols, bf16_t(0.0f));
      for (int b = 0; b < cfg.batch; ++b) {
        for (int q_local = 0; q_local < cfg.seqlen_q; ++q_local) {
          const int q_global = b * cfg.seqlen_q + q_local;
          const int row_kv = cfg.seqlen_k - cfg.seqlen_q + q_local;
          const int row_kv_first = row_kv - (q_local % prefill::kRelBiasQTile);
          const int col_origin = cutlass::fmha::collective::rel_bias_col_origin(
              row_kv_first, cfg.rel_extent, prefill::kRelBiasKTile);
          for (int h = 0; h < cfg.heads_q; ++h) {
            const auto src_base =
                (static_cast<std::size_t>(q_global) * cfg.heads_q + h) * cfg.rel_extent;
            const auto dst_base =
                (static_cast<std::size_t>(q_global) * cfg.heads_q + h) * bias_cols;
            for (int c = 0; c < bias_cols; ++c) {
              const int col = c + col_origin;
              if (col < 0 || col >= cfg.seqlen_k) continue;
              const int rel = row_kv - col;
              if (rel >= 0 && rel < cfg.rel_extent) {
                sheared_rel_bias_host[dst_base + c] = rel_bias_host[src_base + rel];
              }
            }
          }
        }
      }
    }

    DeviceBuffer<bf16_t> q_dev(q, q_host.size());
    DeviceBuffer<bf16_t> k_dev(q, k_host.size());
    DeviceBuffer<bf16_t> v_dev(q, v_host.size());
    DeviceBuffer<bf16_t> out_dev(q, static_cast<std::size_t>(total_q) * cfg.heads_q * cfg.head_dim_v);
    DeviceBuffer<int32_t> cu_q_dev(q, cu_q_host.size());
    DeviceBuffer<int32_t> cu_k_dev(q, cu_k_or_cache_lens_host.size());
    DeviceBuffer<int32_t> page_table_dev;
    DeviceBuffer<bf16_t> sinks_dev;
    DeviceBuffer<bf16_t> rel_bias_dev;

    q_dev.copy_from_host(q_host);
    k_dev.copy_from_host(k_host);
    v_dev.copy_from_host(v_host);
    cu_q_dev.copy_from_host(cu_q_host);
    cu_k_dev.copy_from_host(cu_k_or_cache_lens_host);
    if (cfg.paged) {
      page_table_dev = DeviceBuffer<int32_t>(q, page_table_host.size());
      page_table_dev.copy_from_host(page_table_host);
    }
    if (cfg.sink) {
      sinks_dev = DeviceBuffer<bf16_t>(q, sinks_host.size());
      sinks_dev.copy_from_host(sinks_host);
    }
    if (cfg.relative_bias) {
      rel_bias_dev = DeviceBuffer<bf16_t>(q, sheared_rel_bias_host.size());
      rel_bias_dev.copy_from_host(sheared_rel_bias_host);
    }

    std::cout << "device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";
    std::cout << "shape: batch=" << cfg.batch << " sq=" << cfg.seqlen_q << " sk=" << cfg.seqlen_k
              << " hq=" << cfg.heads_q << " hkv=" << cfg.heads_kv << " d=" << cfg.head_dim
              << " dv=" << cfg.head_dim_v << " paged=" << cfg.paged << " page_size=" << cfg.page_size
              << " causal=" << cfg.causal << " window=(" << cfg.window_left << "," << cfg.window_right
              << ") sink=" << cfg.sink << " relative_bias=" << cfg.relative_bias
              << " rel_extent=" << cfg.rel_extent << "\n";

    auto launch_once = [&] {
      run_prefill(
          cfg,
          q_dev.data(),
          k_dev.data(),
          v_dev.data(),
          out_dev.data(),
          cu_q_dev.data(),
          cu_k_dev.data(),
          cfg.paged ? page_table_dev.data() : nullptr,
          cfg.sink ? sinks_dev.data() : nullptr,
          cfg.relative_bias ? rel_bias_dev.data() : nullptr);
      return sgl_standalone::last_event();
    };

    auto first_event = launch_once();
    first_event.wait();
    q.wait();

    if (cfg.verify) {
      std::vector<float> ref =
          reference_prefill(cfg, q_host, k_host, v_host, page_table_host, sinks_host, rel_bias_host);
      std::vector<bf16_t> out_host = out_dev.copy_to_host();
      if (!verify_output(cfg, out_host, ref)) {
        sgl_standalone::release_workspace();
        return 1;
      }
    }

    for (int i = 0; i < cfg.warmup; ++i) {
      (void)launch_once();
    }
    q.wait();

    // Measure device time over every kernel the prefill call enqueues: a single
    // launch may dispatch more than one kernel, in which case timing only the
    // last event would omit most of the work. clear_events() drops the warmup
    // events so only the measured iterations are summed.
    sgl_standalone::clear_events();
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < cfg.iters; ++i) {
      (void)launch_once();
    }
    q.wait();
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
              << " iters=" << cfg.iters << " estimated_tflops=" << estimate_tflops(cfg, kernel_avg_ms) << "\n";

    sgl_standalone::release_workspace();
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 2;
  }
}
