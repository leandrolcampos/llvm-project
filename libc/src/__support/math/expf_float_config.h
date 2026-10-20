//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Configuration for float-only implementations of expf.
///
/// This file is temporary. It is used to explore different configurations for
/// float_eval::cr_expf_rn.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_MATH_EXPF_FLOAT_CONFIG_H
#define LLVM_LIBC_SRC___SUPPORT_MATH_EXPF_FLOAT_CONFIG_H

#include "src/__support/macros/attributes.h"

// Logical table size. One selects an implementation without a stored table.
#ifndef LIBC_MATH_EXPF_TABLE_SIZE
#define LIBC_MATH_EXPF_TABLE_SIZE 1
#endif

#if LIBC_MATH_EXPF_TABLE_SIZE == 1
#define LIBC_MATH_EXPF_USES_NO_TABLE
#elif LIBC_MATH_EXPF_TABLE_SIZE < 1 || LIBC_MATH_EXPF_TABLE_SIZE > 1024 ||     \
    (LIBC_MATH_EXPF_TABLE_SIZE & (LIBC_MATH_EXPF_TABLE_SIZE - 1)) != 0
#error "Unsupported table size for expf"
#endif

// Maximum number of entries a table can have to be stored in GPU constant
// memory space.
#ifndef LIBC_MATH_EXPF_CONSTANT_TABLE_THRESHOLD
#define LIBC_MATH_EXPF_CONSTANT_TABLE_THRESHOLD 2
#endif

#if defined(LIBC_TARGET_ARCH_IS_GPU) &&                                        \
    LIBC_MATH_EXPF_TABLE_SIZE <= LIBC_MATH_EXPF_CONSTANT_TABLE_THRESHOLD
#define LIBC_MATH_EXPF_TABLE_DECL                                              \
  __attribute__((opencl_constant)) LIBC_INLINE_VAR
#else
#define LIBC_MATH_EXPF_TABLE_DECL LIBC_INLINE_VAR
#endif

// Polynomial evaluation scheme.
#define LIBC_MATH_EXPF_HORNER_SCHEME 1

#ifndef LIBC_MATH_EXPF_POLY_EVAL_SCHEME
#define LIBC_MATH_EXPF_POLY_EVAL_SCHEME LIBC_MATH_EXPF_HORNER_SCHEME
#endif

#if LIBC_MATH_EXPF_POLY_EVAL_SCHEME == LIBC_MATH_EXPF_HORNER_SCHEME
#define LIBC_MATH_EXPF_USES_HORNER_SCHEME
#else
#error "Unsupported polynomial evaluation scheme for expf"
#endif

#endif // LLVM_LIBC_SRC___SUPPORT_MATH_EXPF_FLOAT_CONFIG_H
