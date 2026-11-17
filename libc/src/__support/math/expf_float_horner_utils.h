//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Compensated Horner steps for float-only implementations of expf.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_MATH_EXPF_FLOAT_HORNER_UTILS_H
#define LLVM_LIBC_SRC___SUPPORT_MATH_EXPF_FLOAT_HORNER_UTILS_H

#include "src/__support/FPUtil/double_double.h"
#include "src/__support/FPUtil/multiply_add.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
namespace math {
namespace float_eval_cr_rn {
namespace expf_internal {

using fputil::FloatFloat;

LIBC_ALWAYS_INLINE FloatFloat horner_step(float x, const FloatFloat &poly,
                                          const FloatFloat &coeff) {
  float hi = fputil::multiply_add(x, poly.hi, coeff.hi);
  float lo = fputil::multiply_add(x, poly.hi, coeff.hi - hi);
  lo = fputil::multiply_add(x, poly.lo, lo + coeff.lo);
  return {lo, hi};
}

// The float-only tail has no low word to multiply.
LIBC_ALWAYS_INLINE FloatFloat horner_step(float x, float poly,
                                          const FloatFloat &coeff) {
  float hi = fputil::multiply_add(x, poly, coeff.hi);
  float lo = fputil::multiply_add(x, poly, coeff.hi - hi) + coeff.lo;
  return {lo, hi};
}

// Exact float coefficients do not need a low-word addition.
LIBC_ALWAYS_INLINE FloatFloat horner_step(float x, const FloatFloat &poly,
                                          float coeff) {
  float hi = fputil::multiply_add(x, poly.hi, coeff);
  float lo = fputil::multiply_add(x, poly.hi, coeff - hi);
  lo = fputil::multiply_add(x, poly.lo, lo);
  return {lo, hi};
}

// Neither the float-only tail nor the float coefficient has a low word.
LIBC_ALWAYS_INLINE FloatFloat horner_step(float x, float poly, float coeff) {
  float hi = fputil::multiply_add(x, poly, coeff);
  float lo = fputil::multiply_add(x, poly, coeff - hi);
  return {lo, hi};
}

} // namespace expf_internal
} // namespace float_eval_cr_rn
} // namespace math
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_MATH_EXPF_FLOAT_HORNER_UTILS_H
