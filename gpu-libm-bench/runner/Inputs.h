// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef GPU_LIBM_BENCH_INPUTS_H
#define GPU_LIBM_BENCH_INPUTS_H
#include <cfenv>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace bench {
static_assert(std::numeric_limits<float>::is_iec559 &&
              std::numeric_limits<float>::digits == 24 &&
              std::numeric_limits<double>::digits == 53);
// Host copy of libc/benchmarks/gpu/Random.h's integer algorithm. Keep constants
// and the zero-state fallback identical; no dependency on a configured libc.
class RandomGenerator {
  std::uint64_t state;

public:
  explicit RandomGenerator(std::uint64_t seed) {
    auto x = seed + UINT64_C(0x9E3779B97F4A7C15);
    x = (x ^ (x >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94D049BB133111EB);
    x ^= x >> 31;
    state = x ? x : UINT64_C(0x9E3779B97F4A7C15);
  }
  std::uint32_t next32() {
    auto x = state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    state = x;
    return std::uint32_t((x * UINT64_C(0x2545F4914F6CDD1D)) >> 32);
  }
};
inline std::uint32_t bits(float x) {
  std::uint32_t u;
  std::memcpy(&u, &x, sizeof(u));
  return u;
}
inline std::vector<float> corpus(std::uint64_t seed, std::size_t count) {
  if (!count || (count & (count - 1)) || count > (UINT64_C(1) << 32))
    throw std::invalid_argument("corpus size must be a power of two <= 2^32");
  if (std::fegetround() != FE_TONEAREST)
    throw std::runtime_error("host rounding mode must be RN ties-to-even");
  RandomGenerator rng(seed);
  std::vector<float> result(count);
  for (float &x : result) {
    const std::uint32_t k = rng.next32() >> 8;
    const double u = double(k) * 0x1p-24;
    x = float(-10.0 + 20.0 * u);
  }
  return result;
}
// FNV-1a over little-endian binary32 bytes, independent of native byte order.
inline std::uint64_t fingerprint(const std::vector<float> &input) {
  std::uint64_t h = UINT64_C(0xcbf29ce484222325);
  for (float x : input)
    for (unsigned shift = 0; shift < 32; shift += 8) {
      h ^= (bits(x) >> shift) & 255;
      h *= UINT64_C(0x100000001b3);
    }
  return h;
}
inline std::uint64_t work(std::uint64_t b, std::uint64_t t, std::uint64_t k,
                          std::uint64_t r) {
  std::uint64_t c = 1;
  for (auto v : {b, t, k, r}) {
    if (!v || v > UINT64_MAX / c)
      throw std::invalid_argument("nonpositive or overflowing work count");
    c *= v;
  }
  // NVBench's element count uses signed int64_t.
  if (c > INT64_MAX)
    throw std::invalid_argument("work count exceeds NVBench int64 range");
  return c;
}
} // namespace bench
#endif
