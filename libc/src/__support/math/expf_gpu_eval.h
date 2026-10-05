//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// GPU-optimized implementation of expf.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_MATH_EXPF_GPU_EVAL_H
#define LLVM_LIBC_SRC___SUPPORT_MATH_EXPF_GPU_EVAL_H

#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/PolyEval.h"
#include "src/__support/FPUtil/double_double.h"
#include "src/__support/FPUtil/multiply_add.h"
#include "src/__support/FPUtil/nearest_integer.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/optimization.h"
#include "src/__support/math/exp2f_float_utils.h"

namespace LIBC_NAMESPACE_DECL {
namespace math {
namespace gpu_eval {

namespace expf_internal {

using fputil::FloatFloat;

// Computes a compensated Horner step c + x * p, using the FMA-residual pattern
// and low-part accumulation order from pow_fast.h.
LIBC_INLINE FloatFloat horner_step(float x, const FloatFloat &p,
                                   const FloatFloat &c) {
  float hi = fputil::multiply_add(x, p.hi, c.hi);
  float lo = fputil::multiply_add(x, p.hi, c.hi - hi);
  lo = fputil::multiply_add(x, p.lo, lo + c.lo);
  return {lo, hi};
}

// Approximates exp(u) with the degree-11 polynomial P(u) = 1 + u + u^2 * Q(u),
// evaluated using Horner's scheme.
LIBC_INLINE FloatFloat exp_eval(const FloatFloat &u) {
  // Coefficients of u^2, ..., u^7, in {lo, hi} order.
  constexpr FloatFloat COEFFS[] = {
      {0x1p-49f, 0x1p-1f},
      {-0x1.555558p-28f, 0x1.555556p-3f},
      {-0x1.556236p-30f, 0x1.555556p-5f},
      {-0x1.ddd2aep-32f, 0x1.111112p-7f},
      {-0x1.dd73d4p-36f, 0x1.6c16c2p-10f},
      {-0x1.ec0c88p-39f, 0x1.a01a02p-13f},
  };

  // The float-only tail contributes to degrees 8 through 11 of P.
  float tail = fputil::polyeval(u.hi, 0x1.a0199p-16f, 0x1.71de5ep-19f,
                                0x1.28b56ep-22f, 0x1.aeb702p-26f);
  FloatFloat p{0.0f, tail};
  for (int i = 5; i >= 0; --i)
    p = horner_step(u.hi, p, COEFFS[i]);

  // Form expm1(u.hi) before adding 1, retaining its low component.
  p = horner_step(u.hi, p, FloatFloat{0.0f, 1.0f});
  p = fputil::quick_mult(u.hi, p);
  FloatFloat y = fputil::exact_add(1.0f, p.hi);

  // exp(u.hi + u.lo) = exp(u.hi) + u.lo * exp(u.hi) + O(u.lo^2).
  //
  // Apply this correction once, rather than propagating u.lo through every
  // Horner step.
  p.lo = fputil::multiply_add(u.lo, y.hi, p.lo);

  // Rounding y.lo + p.lo to nearest can create a midpoint for y.hi + y.lo.
  // Round inexact low sums to odd whenever their rounded value is a power
  // of two. This covers all such midpoint cases in this evaluation and
  // preserves which side of the midpoint the exact sum lies on.
  //
  // CORE-MATH's as_exp_accurate uses a related residual-directed adjustment
  // of the low component before the final summation.
  //
  // FastTwoSum is exact in RN: for nonzero p.lo, y.lo is an integer multiple
  // of ulp(p.lo), as required by Theorem 1 of "FastTwoSum revisited".
  FloatFloat lo = fputil::exact_add(y.lo, p.lo);

  uint32_t lo_bits = fputil::FPBits<float>(lo.hi).uintval();
  if ((lo_bits & fputil::FPBits<float>::FRACTION_MASK) == 0 && lo.lo != 0.0f)
    lo_bits += ((lo.hi > 0.0f) == (lo.lo > 0.0f)) ? 1U : uint32_t(-1);
  y.lo = fputil::FPBits<float>(lo_bits).get_val();
  return y;
}

// Reconstructs a subnormal result from y * 2^k in round-to-nearest mode.
// The caller guarantees -150 <= k <= -126 and y * 2^k < 2^-126.
LIBC_INLINE float scale_subnormal(const FloatFloat &y, int k) {
  using FPBits = fputil::FPBits<float>;

  // Adding 2^(-126-k) aligns the rounding grid with subnormal spacing:
  // ulp(offset) * 2^k = 2^-149.
  uint32_t offset_u = static_cast<uint32_t>(1 - k) << FPBits::FRACTION_LEN;
  float offset = FPBits(offset_u).get_val();
  FloatFloat sum = fputil::exact_add(offset, y.hi);
  float rounded = sum.hi + (sum.lo + y.lo);

  return FPBits(FPBits(rounded).uintval() - offset_u).get_val();
}

} // namespace expf_internal

LIBC_INLINE float expf(float x) {
  using FPBits = fputil::FPBits<float>;
  FPBits x_bits(x);

  uint32_t x_u = x_bits.uintval();
  uint32_t x_abs_u = x_u & 0x7fff'ffffU;

  // When |x| >= 89, |x| < 2^-25, or x is nan
  if (LIBC_UNLIKELY(x_abs_u >= 0x42b2'0000U || x_abs_u <= 0x3300'0000U)) {
    // |x| < 2^-25
    if (x_abs_u <= 0x3300'0000U)
      return 1.0f;

    // When x <= log(2^-150) or nan
    if (x_u >= 0xc2cf'f1b5U) {
      if (x_bits.is_nan())
        return x;

      return 0.0f;
    }

    // x >= 89 or nan
    if (x_bits.is_pos() && (x_u >= 0x42b2'0000U))
      return x + FPBits::inf().get_val();
  }

  // Constants generated by Sollya with:
  //   > display = hexadecimal;
  //   > LOG2_E = round(log2(exp(1)), SG, RN);
  //   > LOG_2_HI = round(log(2), SG, RN);
  //   > LOG_2_LO.hi = round(log(2) - LOG_2_HI, SG, RN);
  //   > LOG_2_LO.lo = round(log(2) - LOG_2_HI - LOG_2_LO.hi, SG, RN);
  constexpr float LOG2_E = 0x1.715476p+0f;
  constexpr float LOG_2_HI = 0x1.62e43p-1f;
  constexpr fputil::FloatFloat LOG_2_LO = {/*lo=*/-0x1.950d88p-54f,
                                           /*hi=*/-0x1.05c61p-29f};

  // Range reduction:
  //   k = round(x * log2(e))
  //   x * log2(e) = k + u * log2(e)
  //   e^x = 2^k * e^u
  float kf = fputil::nearest_integer(x * LOG2_E);
  int k = static_cast<int>(kf);

  float u_hi = fputil::multiply_add(-kf, LOG_2_HI, x);
  fputil::FloatFloat u_lo = fputil::quick_mult(-kf, LOG_2_LO);

  fputil::FloatFloat u = fputil::exact_add(u_hi, u_lo.hi);
  u.lo += u_lo.lo;

  fputil::FloatFloat y = expf_internal::exp_eval(u);

  if (LIBC_UNLIKELY(x_u >= 0xc2ae'ac50U))
    return expf_internal::scale_subnormal(y, k);

  return float_eval::scale_exp2f(y.hi + y.lo, k);
}

} // namespace gpu_eval
} // namespace math
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_MATH_EXPF_GPU_EVAL_H
