// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef GPU_LIBM_BENCH_MEASUREMENT_H
#define GPU_LIBM_BENCH_MEASUREMENT_H
#include <cmath>
#include <cstdint>

namespace bench {
struct Measurement {
  double seconds = 0, wall_seconds = 0, target = 0, timeout = 0;
  std::int64_t launches = 0, requested_minimum = 0, effective_minimum = 0;
  bool has_batch = false, skipped = false, failed = false;

  // This classification is independent of the chosen output format.
  int code() const {
    if (failed || !std::isfinite(target) || target <= 0 ||
        !std::isfinite(timeout) || timeout <= 0 || requested_minimum < 1)
      return 1;
    if (skipped || !has_batch || !std::isfinite(seconds) || seconds <= 0 ||
        launches < effective_minimum || seconds * launches < target)
      return 2;
    return 0;
  }
};
} // namespace bench
#endif
