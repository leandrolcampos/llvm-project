// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Compiled without CUDA headers; all measured candidates have a C ABI.
extern "C" __attribute__((device)) float BENCH_SYMBOL(float);
using u32 = unsigned;
using u64 = unsigned long long;
static_assert(sizeof(float) == sizeof(u32));

__attribute__((device, always_inline)) inline u32 thread_id() {
  return __nvvm_read_ptx_sreg_ctaid_x() * __nvvm_read_ptx_sreg_ntid_x() +
         __nvvm_read_ptx_sreg_tid_x();
}

template <int K>
__attribute__((device, always_inline)) inline void
evaluate(const float *input, u32 *output, u64 repeats, u32 mask) {
  u32 tid = thread_id();
  u32 args[K], checksums[K] = {};
#pragma unroll
  for (int k = 0; k < K; ++k)
    args[k] = __builtin_bit_cast(u32, input[(u64(tid) * K + k) & mask]);
#pragma clang loop unroll(disable)
  for (u64 r = 0; r < repeats; ++r) {
#pragma unroll
    for (int k = 0; k < K; ++k) {
      // Actual, opaque-to-LLVM PTX input update. Sign alternation preserves
      // [-10,10], varies arguments identically for every candidate, and makes
      // repeated evaluations non-invariant. Audit ptxas output as well.
      asm volatile("xor.b32 %0, %0, 0x80000000;" : "+r"(args[k]));
      float y = BENCH_SYMBOL(__builtin_bit_cast(float, args[k]));
      // Every evaluation contributes to a separate observable integer stream.
      checksums[k] = ((checksums[k] << 1) | (checksums[k] >> 31)) +
                     __builtin_bit_cast(u32, y);
    }
  }
#pragma unroll
  for (int k = 0; k < K; ++k)
    output[u64(tid) * K + k] = checksums[k];
}

#define KERNEL(K)                                                              \
  extern "C" __attribute__((global)) void expf_k##K(                           \
      const float *input, u32 *output, u64 repeats, u32 mask) {                \
    evaluate<K>(input, output, repeats, mask);                                 \
  }
BENCH_KERNELS

extern "C" __attribute__((global)) void expf_probe(const float *input,
                                                   float *output, u32 count) {
  u32 tid = thread_id();
  if (tid < count)
    output[tid] = BENCH_SYMBOL(input[tid]);
}
