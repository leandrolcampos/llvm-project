// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef GPU_LIBM_BENCH_VALIDATION_H
#define GPU_LIBM_BENCH_VALIDATION_H
#include <exception>
#include <map>

namespace bench {
// A successful check runs once. A failed check permanently rejects its key,
// preserving the original exception for later states in the same invocation.
template <typename Key> class ValidationCache {
  std::map<Key, std::exception_ptr> results;

public:
  template <typename Check> void ensure(const Key &key, Check &&check) {
    auto previous = results.find(key);
    if (previous != results.end()) {
      if (previous->second)
        std::rethrow_exception(previous->second);
      return;
    }
    try {
      check();
    } catch (...) {
      results.emplace(key, std::current_exception());
      throw;
    }
    results.emplace(key, nullptr);
  }
};
} // namespace bench
#endif
