//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Utilities for float-only implementations of expf.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_MATH_EXPF_FLOAT_UTILS_H
#define LLVM_LIBC_SRC___SUPPORT_MATH_EXPF_FLOAT_UTILS_H

#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/double_double.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/math/expf_float_config.h"

#ifdef LIBC_MATH_EXPF_USES_HORNER_SCHEME
#include "src/__support/math/expf_float_horner.h"
#endif

namespace LIBC_NAMESPACE_DECL {
namespace math {
namespace float_eval {

namespace cr_expf_rn_internal {

using fputil::FloatFloat;

// Computes 2^k * (exp_u.hi + exp_u.lo).
LIBC_INLINE float scale_result(const FloatFloat &exp_u, int k,
                               bool produces_subnormal) {
  using FPBits = fputil::FPBits<float>;

  // Scaling the rounded sum by 2^k is exact for normal results: only the
  // exponent changes. For subnormal results, the fixed spacing 2^-149 is
  // coarser than the scaled spacing of the rounded sum. Scaling can then
  // discard significand bits and round again.
  //
  // Choose B = 2^(-126 - k), so ulp(B) * 2^k = 2^-149. For inputs producing
  // subnormals, B + exp_u.hi + exp_u.lo lies in B's binade [B, 2B). Rounding
  // this biased sum therefore uses the desired spacing while staying normal.
  // Use B = 0 for normal results to share the same arithmetic path.
  uint32_t bias_u = produces_subnormal ? (1U - static_cast<uint32_t>(k))
                                             << FPBits::FRACTION_LEN
                                       : 0U;
  float bias = FPBits(bias_u).get_val();

  // For inputs producing subnormals, B >= exp_u.hi, so FastTwoSum retains
  // the rounding residual exactly in RN. It is also exact when B = 0,
  // reducing the final addition to RN(exp_u.hi + exp_u.lo).
  FloatFloat sum_biased = fputil::exact_add(bias, exp_u.hi);
  float sum_rounded = sum_biased.hi + (sum_biased.lo + exp_u.lo);
  uint32_t res_u = FPBits(sum_rounded).uintval();

  // For subnormal results, sum_rounded lies in B's binade [B, 2B):
  // sum_rounded = B + M * ulp(B), where M is the integer encoded by
  // its fraction field. Clearing the exponent makes the same fraction
  // bits represent M * 2^-149 = (sum_rounded - B) * 2^k. This removes
  // the bias and applies the scale without further rounding.
  uint32_t res_mask = produces_subnormal ? FPBits::FRACTION_MASK : ~0U;
  uint32_t exponent_adj = produces_subnormal ? 0U
                                             : static_cast<uint32_t>(k)
                                                   << FPBits::FRACTION_LEN;
  return FPBits((res_u & res_mask) + exponent_adj).get_val();
}

} // namespace cr_expf_rn_internal

} // namespace float_eval
} // namespace math
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_MATH_EXPF_FLOAT_UTILS_H
