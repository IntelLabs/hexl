// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "hexl/eltwise/eltwise-add-mod.hpp"

#include "eltwise/eltwise-add-mod-avx512.hpp"
#include "eltwise/eltwise-add-mod-internal.hpp"
#include "hexl/logging/logging.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "hexl/util/check.hpp"
#include "util/cpu-features.hpp"
//#include <omp.h>

namespace intel {
namespace hexl {

void EltwiseAddModNative(uint64_t* result, const uint64_t* operand1,
                         const uint64_t* operand2, uint64_t n,
                         uint64_t modulus) {
  HEXL_CHECK(result != nullptr, "Require result != nullptr");
  HEXL_CHECK(operand1 != nullptr, "Require operand1 != nullptr");
  HEXL_CHECK(operand2 != nullptr, "Require operand2 != nullptr");
  HEXL_CHECK(n != 0, "Require n != 0");
  HEXL_CHECK(modulus > 1, "Require modulus > 1");
  HEXL_CHECK(modulus < (1ULL << 63), "Require modulus < 2**63");
  HEXL_CHECK_BOUNDS(operand1, n, modulus,
                    "pre-add value in operand1 exceeds bound " << modulus);
  HEXL_CHECK_BOUNDS(operand2, n, modulus,
                    "pre-add value in operand2 exceeds bound " << modulus);

//  HEXL_LOOP_UNROLL_4
  for (size_t i = 0; i < n; ++i) {
    uint64_t sum = *operand1 + *operand2;
    if (sum >= modulus) {
      *result = sum - modulus;
    } else {
      *result = sum;
    }

    ++operand1;
    ++operand2;
    ++result;
  }


//#pragma omp parallel
//{
//  size_t i;
//    const uint64_t* local_operand1;
//    const uint64_t* local_operand2;
//  uint64_t* local_result;
//
//  #pragma omp for private(i, local_operand1, local_operand2, local_result)
//  for (i = 0; i < n; ++i) {
//    local_operand1 = operand1 + i;
//    local_operand2 = operand2 + i;
//    local_result = result + i;
//
//    uint64_t sum = *local_operand1 + *local_operand2;
//    if (sum >= modulus) {
//      *local_result = sum - modulus;
//    } else {
//      *local_result = sum;
//    }
//  }
//}

}

void EltwiseAddModNative(uint64_t* result, const uint64_t* operand1,
                         uint64_t operand2, uint64_t n, uint64_t modulus) {
  HEXL_CHECK(result != nullptr, "Require result != nullptr");
  HEXL_CHECK(operand1 != nullptr, "Require operand1 != nullptr");
  HEXL_CHECK(n != 0, "Require n != 0");
  HEXL_CHECK(modulus > 1, "Require modulus > 1");
  HEXL_CHECK(modulus < (1ULL << 63), "Require modulus < 2**63");
  HEXL_CHECK_BOUNDS(operand1, n, modulus,
                    "pre-add value in operand1 exceeds bound " << modulus);
  HEXL_CHECK(operand2 < modulus, "Require operand2 < modulus");

  uint64_t diff = modulus - operand2;

//  HEXL_LOOP_UNROLL_4
  for (size_t i = 0; i < n; ++i) {
    if (*operand1 >= diff) {
      *result = *operand1 - diff;
    } else {
      *result = *operand1 + operand2;
    }

    ++operand1;
    ++result;
  }
    
//    #pragma omp parallel
//    {
//      size_t i;
//      const uint64_t* local_operand1;
//      uint64_t* local_result;
//
//      #pragma omp for private(i, local_operand1, local_result)
//      for (i = 0; i < n; ++i) {
//        local_operand1 = operand1 + i;
//        local_result = result + i;
//
//        if (*local_operand1 >= diff) {
//          *local_result = *local_operand1 - diff;
//        } else {
//          *local_result = *local_operand1 + operand2;
//        }
//      }
//    }
}

void EltwiseAddMod(uint64_t* result, const uint64_t* operand1,
                   const uint64_t* operand2, uint64_t n, uint64_t modulus) {
  HEXL_CHECK(operand1 != nullptr, "Require operand1 != nullptr");
  HEXL_CHECK(operand2 != nullptr, "Require operand2 != nullptr");
  HEXL_CHECK(n != 0, "Require n != 0");
  HEXL_CHECK(modulus > 1, "Require modulus > 1");
  HEXL_CHECK(modulus < (1ULL << 63), "Require modulus < 2**63");
  HEXL_CHECK_BOUNDS(operand1, n, modulus,
                    "pre-add value in operand1 exceeds bound " << modulus);
  HEXL_CHECK_BOUNDS(operand2, n, modulus,
                    "pre-add value in operand2 exceeds bound " << modulus);

#ifdef HEXL_HAS_AVX512DQ
  if (has_avx512dq) {
    EltwiseAddModAVX512(result, operand1, operand2, n, modulus);
    return;
  }
#endif

  HEXL_VLOG(3, "Calling EltwiseAddModNative");
  EltwiseAddModNative(result, operand1, operand2, n, modulus);
}

void EltwiseAddMod(uint64_t* result, const uint64_t* operand1,
                   const uint64_t operand2, uint64_t n, uint64_t modulus) {
  HEXL_CHECK(operand1 != nullptr, "Require operand1 != nullptr");
  HEXL_CHECK(n != 0, "Require n != 0");
  HEXL_CHECK(modulus > 1, "Require modulus > 1");
  HEXL_CHECK(modulus < (1ULL << 63), "Require modulus < 2**63");
  HEXL_CHECK_BOUNDS(operand1, n, modulus,
                    "pre-add value in operand1 exceeds bound " << modulus);
  HEXL_CHECK(operand2 < modulus, "Require operand2 < modulus");

#ifdef HEXL_HAS_AVX512DQ
  if (has_avx512dq) {
    EltwiseAddModAVX512(result, operand1, operand2, n, modulus);
    return;
  }
#endif

  HEXL_VLOG(3, "Calling EltwiseAddModNative");
  EltwiseAddModNative(result, operand1, operand2, n, modulus);
}

}  // namespace hexl
}  // namespace intel
