// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "eltwise/eltwise-mult-mod-avx512.hpp"

#include <immintrin.h>
#include <stdint.h>

#include <limits>

#include "eltwise/eltwise-mult-mod-internal.hpp"
#include "hexl/eltwise/eltwise-mult-mod.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "hexl/util/check.hpp"
#include "hexl/util/compiler.hpp"
#include "hexl/util/defines.hpp"
#include "util/avx512-util.hpp"

namespace intel {
namespace hexl {

#ifdef HEXL_HAS_AVX512DQ

template void EltwiseMultModAVX512Float<1>(uint64_t* result,
                                           const uint64_t* operand1,
                                           const uint64_t* operand2, uint64_t n,
                                           uint64_t modulus);
template void EltwiseMultModAVX512Float<2>(uint64_t* result,
                                           const uint64_t* operand1,
                                           const uint64_t* operand2, uint64_t n,
                                           uint64_t modulus);
template void EltwiseMultModAVX512Float<4>(uint64_t* result,
                                           const uint64_t* operand1,
                                           const uint64_t* operand2, uint64_t n,
                                           uint64_t modulus);

template void EltwiseMultModAVX512Int<1>(uint64_t* result,
                                         const uint64_t* operand1,
                                         const uint64_t* operand2, uint64_t n,
                                         uint64_t modulus);
template void EltwiseMultModAVX512Int<2>(uint64_t* result,
                                         const uint64_t* operand1,
                                         const uint64_t* operand2, uint64_t n,
                                         uint64_t modulus);
template void EltwiseMultModAVX512Int<4>(uint64_t* result,
                                         const uint64_t* operand1,
                                         const uint64_t* operand2, uint64_t n,
                                         uint64_t modulus);

#endif

#ifdef HEXL_HAS_AVX512DQ

template <int BitShift, int InputModFactor, int CoeffCount>
void EltwiseMultModAVX512IntLoopUnroll(__m512i* vp_result,
                                       const __m512i* vp_operand1,
                                       const __m512i* vp_operand2,
                                       __m512i vbarr_lo, __m512i v_modulus,
                                       __m512i v_twice_mod) {
  constexpr size_t manual_unroll_factor = 16;
  constexpr size_t avx512_64bit_count = 8;
  constexpr size_t loop_count =
      CoeffCount / (manual_unroll_factor * avx512_64bit_count);

  static_assert(loop_count > 0, "loop_count too small for unrolling");
  static_assert(CoeffCount % (manual_unroll_factor * avx512_64bit_count) == 0,
                "CoeffCount must be a factor of manual_unroll_factor * "
                "avx512_64bit_count");

  HEXL_UNUSED(v_twice_mod);
  HEXL_LOOP_UNROLL_4
  for (size_t i = loop_count; i > 0; --i) {
    __m512i x1 = _mm512_loadu_si512(vp_operand1++);
    __m512i y1 = _mm512_loadu_si512(vp_operand2++);
    __m512i x2 = _mm512_loadu_si512(vp_operand1++);
    __m512i y2 = _mm512_loadu_si512(vp_operand2++);
    __m512i x3 = _mm512_loadu_si512(vp_operand1++);
    __m512i y3 = _mm512_loadu_si512(vp_operand2++);
    __m512i x4 = _mm512_loadu_si512(vp_operand1++);
    __m512i y4 = _mm512_loadu_si512(vp_operand2++);
    __m512i x5 = _mm512_loadu_si512(vp_operand1++);
    __m512i y5 = _mm512_loadu_si512(vp_operand2++);
    __m512i x6 = _mm512_loadu_si512(vp_operand1++);
    __m512i y6 = _mm512_loadu_si512(vp_operand2++);
    __m512i x7 = _mm512_loadu_si512(vp_operand1++);
    __m512i y7 = _mm512_loadu_si512(vp_operand2++);
    __m512i x8 = _mm512_loadu_si512(vp_operand1++);
    __m512i y8 = _mm512_loadu_si512(vp_operand2++);
    __m512i x9 = _mm512_loadu_si512(vp_operand1++);
    __m512i y9 = _mm512_loadu_si512(vp_operand2++);
    __m512i x10 = _mm512_loadu_si512(vp_operand1++);
    __m512i y10 = _mm512_loadu_si512(vp_operand2++);
    __m512i x11 = _mm512_loadu_si512(vp_operand1++);
    __m512i y11 = _mm512_loadu_si512(vp_operand2++);
    __m512i x12 = _mm512_loadu_si512(vp_operand1++);
    __m512i y12 = _mm512_loadu_si512(vp_operand2++);
    __m512i x13 = _mm512_loadu_si512(vp_operand1++);
    __m512i y13 = _mm512_loadu_si512(vp_operand2++);
    __m512i x14 = _mm512_loadu_si512(vp_operand1++);
    __m512i y14 = _mm512_loadu_si512(vp_operand2++);
    __m512i x15 = _mm512_loadu_si512(vp_operand1++);
    __m512i y15 = _mm512_loadu_si512(vp_operand2++);
    __m512i x16 = _mm512_loadu_si512(vp_operand1++);
    __m512i y16 = _mm512_loadu_si512(vp_operand2++);

    x1 = _mm512_hexl_small_mod_epu64<InputModFactor>(x1, v_modulus,
                                                     &v_twice_mod);
    x2 = _mm512_hexl_small_mod_epu64<InputModFactor>(x2, v_modulus,
                                                     &v_twice_mod);
    x3 = _mm512_hexl_small_mod_epu64<InputModFactor>(x3, v_modulus,
                                                     &v_twice_mod);
    x4 = _mm512_hexl_small_mod_epu64<InputModFactor>(x4, v_modulus,
                                                     &v_twice_mod);
    x5 = _mm512_hexl_small_mod_epu64<InputModFactor>(x5, v_modulus,
                                                     &v_twice_mod);
    x6 = _mm512_hexl_small_mod_epu64<InputModFactor>(x6, v_modulus,
                                                     &v_twice_mod);
    x7 = _mm512_hexl_small_mod_epu64<InputModFactor>(x7, v_modulus,
                                                     &v_twice_mod);
    x8 = _mm512_hexl_small_mod_epu64<InputModFactor>(x8, v_modulus,
                                                     &v_twice_mod);
    x9 = _mm512_hexl_small_mod_epu64<InputModFactor>(x9, v_modulus,
                                                     &v_twice_mod);
    x10 = _mm512_hexl_small_mod_epu64<InputModFactor>(x10, v_modulus,
                                                      &v_twice_mod);
    x11 = _mm512_hexl_small_mod_epu64<InputModFactor>(x11, v_modulus,
                                                      &v_twice_mod);
    x12 = _mm512_hexl_small_mod_epu64<InputModFactor>(x12, v_modulus,
                                                      &v_twice_mod);
    x13 = _mm512_hexl_small_mod_epu64<InputModFactor>(x13, v_modulus,
                                                      &v_twice_mod);
    x14 = _mm512_hexl_small_mod_epu64<InputModFactor>(x14, v_modulus,
                                                      &v_twice_mod);
    x15 = _mm512_hexl_small_mod_epu64<InputModFactor>(x15, v_modulus,
                                                      &v_twice_mod);
    x16 = _mm512_hexl_small_mod_epu64<InputModFactor>(x16, v_modulus,
                                                      &v_twice_mod);

    y1 = _mm512_hexl_small_mod_epu64<InputModFactor>(y1, v_modulus,
                                                     &v_twice_mod);
    y2 = _mm512_hexl_small_mod_epu64<InputModFactor>(y2, v_modulus,
                                                     &v_twice_mod);
    y3 = _mm512_hexl_small_mod_epu64<InputModFactor>(y3, v_modulus,
                                                     &v_twice_mod);
    y4 = _mm512_hexl_small_mod_epu64<InputModFactor>(y4, v_modulus,
                                                     &v_twice_mod);
    y5 = _mm512_hexl_small_mod_epu64<InputModFactor>(y5, v_modulus,
                                                     &v_twice_mod);
    y6 = _mm512_hexl_small_mod_epu64<InputModFactor>(y6, v_modulus,
                                                     &v_twice_mod);
    y7 = _mm512_hexl_small_mod_epu64<InputModFactor>(y7, v_modulus,
                                                     &v_twice_mod);
    y8 = _mm512_hexl_small_mod_epu64<InputModFactor>(y8, v_modulus,
                                                     &v_twice_mod);
    y9 = _mm512_hexl_small_mod_epu64<InputModFactor>(y9, v_modulus,
                                                     &v_twice_mod);
    y10 = _mm512_hexl_small_mod_epu64<InputModFactor>(y10, v_modulus,
                                                      &v_twice_mod);
    y11 = _mm512_hexl_small_mod_epu64<InputModFactor>(y11, v_modulus,
                                                      &v_twice_mod);
    y12 = _mm512_hexl_small_mod_epu64<InputModFactor>(y12, v_modulus,
                                                      &v_twice_mod);
    y13 = _mm512_hexl_small_mod_epu64<InputModFactor>(y13, v_modulus,
                                                      &v_twice_mod);
    y14 = _mm512_hexl_small_mod_epu64<InputModFactor>(y14, v_modulus,
                                                      &v_twice_mod);
    y15 = _mm512_hexl_small_mod_epu64<InputModFactor>(y15, v_modulus,
                                                      &v_twice_mod);
    y16 = _mm512_hexl_small_mod_epu64<InputModFactor>(y16, v_modulus,
                                                      &v_twice_mod);

    __m512i zhi1 = _mm512_hexl_mulhi_epi<64>(x1, y1);
    __m512i zhi2 = _mm512_hexl_mulhi_epi<64>(x2, y2);
    __m512i zhi3 = _mm512_hexl_mulhi_epi<64>(x3, y3);
    __m512i zhi4 = _mm512_hexl_mulhi_epi<64>(x4, y4);
    __m512i zhi5 = _mm512_hexl_mulhi_epi<64>(x5, y5);
    __m512i zhi6 = _mm512_hexl_mulhi_epi<64>(x6, y6);
    __m512i zhi7 = _mm512_hexl_mulhi_epi<64>(x7, y7);
    __m512i zhi8 = _mm512_hexl_mulhi_epi<64>(x8, y8);
    __m512i zhi9 = _mm512_hexl_mulhi_epi<64>(x9, y9);
    __m512i zhi10 = _mm512_hexl_mulhi_epi<64>(x10, y10);
    __m512i zhi11 = _mm512_hexl_mulhi_epi<64>(x11, y11);
    __m512i zhi12 = _mm512_hexl_mulhi_epi<64>(x12, y12);
    __m512i zhi13 = _mm512_hexl_mulhi_epi<64>(x13, y13);
    __m512i zhi14 = _mm512_hexl_mulhi_epi<64>(x14, y14);
    __m512i zhi15 = _mm512_hexl_mulhi_epi<64>(x15, y15);
    __m512i zhi16 = _mm512_hexl_mulhi_epi<64>(x16, y16);

    __m512i zlo1 = _mm512_hexl_mullo_epi<64>(x1, y1);
    __m512i zlo2 = _mm512_hexl_mullo_epi<64>(x2, y2);
    __m512i zlo3 = _mm512_hexl_mullo_epi<64>(x3, y3);
    __m512i zlo4 = _mm512_hexl_mullo_epi<64>(x4, y4);
    __m512i zlo5 = _mm512_hexl_mullo_epi<64>(x5, y5);
    __m512i zlo6 = _mm512_hexl_mullo_epi<64>(x6, y6);
    __m512i zlo7 = _mm512_hexl_mullo_epi<64>(x7, y7);
    __m512i zlo8 = _mm512_hexl_mullo_epi<64>(x8, y8);
    __m512i zlo9 = _mm512_hexl_mullo_epi<64>(x9, y9);
    __m512i zlo10 = _mm512_hexl_mullo_epi<64>(x10, y10);
    __m512i zlo11 = _mm512_hexl_mullo_epi<64>(x11, y11);
    __m512i zlo12 = _mm512_hexl_mullo_epi<64>(x12, y12);
    __m512i zlo13 = _mm512_hexl_mullo_epi<64>(x13, y13);
    __m512i zlo14 = _mm512_hexl_mullo_epi<64>(x14, y14);
    __m512i zlo15 = _mm512_hexl_mullo_epi<64>(x15, y15);
    __m512i zlo16 = _mm512_hexl_mullo_epi<64>(x16, y16);

    __m512i c1 = _mm512_hexl_shrdi_epi64<BitShift - 1>(zlo1, zhi1);
    __m512i c2 = _mm512_hexl_shrdi_epi64<BitShift - 1>(zlo2, zhi2);
    __m512i c3 = _mm512_hexl_shrdi_epi64<BitShift - 1>(zlo3, zhi3);
    __m512i c4 = _mm512_hexl_shrdi_epi64<BitShift - 1>(zlo4, zhi4);
    __m512i c5 = _mm512_hexl_shrdi_epi64<BitShift - 1>(zlo5, zhi5);
    __m512i c6 = _mm512_hexl_shrdi_epi64<BitShift - 1>(zlo6, zhi6);
    __m512i c7 = _mm512_hexl_shrdi_epi64<BitShift - 1>(zlo7, zhi7);
    __m512i c8 = _mm512_hexl_shrdi_epi64<BitShift - 1>(zlo8, zhi8);
    __m512i c9 = _mm512_hexl_shrdi_epi64<BitShift - 1>(zlo9, zhi9);
    __m512i c10 = _mm512_hexl_shrdi_epi64<BitShift - 1>(zlo10, zhi10);
    __m512i c11 = _mm512_hexl_shrdi_epi64<BitShift - 1>(zlo11, zhi11);
    __m512i c12 = _mm512_hexl_shrdi_epi64<BitShift - 1>(zlo12, zhi12);
    __m512i c13 = _mm512_hexl_shrdi_epi64<BitShift - 1>(zlo13, zhi13);
    __m512i c14 = _mm512_hexl_shrdi_epi64<BitShift - 1>(zlo14, zhi14);
    __m512i c15 = _mm512_hexl_shrdi_epi64<BitShift - 1>(zlo15, zhi15);
    __m512i c16 = _mm512_hexl_shrdi_epi64<BitShift - 1>(zlo16, zhi16);

    c1 = _mm512_hexl_mulhi_approx_epi<64>(c1, vbarr_lo);
    c2 = _mm512_hexl_mulhi_approx_epi<64>(c2, vbarr_lo);
    c3 = _mm512_hexl_mulhi_approx_epi<64>(c3, vbarr_lo);
    c4 = _mm512_hexl_mulhi_approx_epi<64>(c4, vbarr_lo);
    c5 = _mm512_hexl_mulhi_approx_epi<64>(c5, vbarr_lo);
    c6 = _mm512_hexl_mulhi_approx_epi<64>(c6, vbarr_lo);
    c7 = _mm512_hexl_mulhi_approx_epi<64>(c7, vbarr_lo);
    c8 = _mm512_hexl_mulhi_approx_epi<64>(c8, vbarr_lo);
    c9 = _mm512_hexl_mulhi_approx_epi<64>(c9, vbarr_lo);
    c10 = _mm512_hexl_mulhi_approx_epi<64>(c10, vbarr_lo);
    c11 = _mm512_hexl_mulhi_approx_epi<64>(c11, vbarr_lo);
    c12 = _mm512_hexl_mulhi_approx_epi<64>(c12, vbarr_lo);
    c13 = _mm512_hexl_mulhi_approx_epi<64>(c13, vbarr_lo);
    c14 = _mm512_hexl_mulhi_approx_epi<64>(c14, vbarr_lo);
    c15 = _mm512_hexl_mulhi_approx_epi<64>(c15, vbarr_lo);
    c16 = _mm512_hexl_mulhi_approx_epi<64>(c16, vbarr_lo);

    __m512i vr1 = _mm512_hexl_mullo_epi<64>(c1, v_modulus);
    __m512i vr2 = _mm512_hexl_mullo_epi<64>(c2, v_modulus);
    __m512i vr3 = _mm512_hexl_mullo_epi<64>(c3, v_modulus);
    __m512i vr4 = _mm512_hexl_mullo_epi<64>(c4, v_modulus);
    __m512i vr5 = _mm512_hexl_mullo_epi<64>(c5, v_modulus);
    __m512i vr6 = _mm512_hexl_mullo_epi<64>(c6, v_modulus);
    __m512i vr7 = _mm512_hexl_mullo_epi<64>(c7, v_modulus);
    __m512i vr8 = _mm512_hexl_mullo_epi<64>(c8, v_modulus);
    __m512i vr9 = _mm512_hexl_mullo_epi<64>(c9, v_modulus);
    __m512i vr10 = _mm512_hexl_mullo_epi<64>(c10, v_modulus);
    __m512i vr11 = _mm512_hexl_mullo_epi<64>(c11, v_modulus);
    __m512i vr12 = _mm512_hexl_mullo_epi<64>(c12, v_modulus);
    __m512i vr13 = _mm512_hexl_mullo_epi<64>(c13, v_modulus);
    __m512i vr14 = _mm512_hexl_mullo_epi<64>(c14, v_modulus);
    __m512i vr15 = _mm512_hexl_mullo_epi<64>(c15, v_modulus);
    __m512i vr16 = _mm512_hexl_mullo_epi<64>(c16, v_modulus);

    vr1 = _mm512_sub_epi64(zlo1, vr1);
    vr2 = _mm512_sub_epi64(zlo2, vr2);
    vr3 = _mm512_sub_epi64(zlo3, vr3);
    vr4 = _mm512_sub_epi64(zlo4, vr4);
    vr5 = _mm512_sub_epi64(zlo5, vr5);
    vr6 = _mm512_sub_epi64(zlo6, vr6);
    vr7 = _mm512_sub_epi64(zlo7, vr7);
    vr8 = _mm512_sub_epi64(zlo8, vr8);
    vr9 = _mm512_sub_epi64(zlo9, vr9);
    vr10 = _mm512_sub_epi64(zlo10, vr10);
    vr11 = _mm512_sub_epi64(zlo11, vr11);
    vr12 = _mm512_sub_epi64(zlo12, vr12);
    vr13 = _mm512_sub_epi64(zlo13, vr13);
    vr14 = _mm512_sub_epi64(zlo14, vr14);
    vr15 = _mm512_sub_epi64(zlo15, vr15);
    vr16 = _mm512_sub_epi64(zlo16, vr16);

    vr1 = _mm512_hexl_small_mod_epu64<4>(vr1, v_modulus, &v_twice_mod);
    vr2 = _mm512_hexl_small_mod_epu64<4>(vr2, v_modulus, &v_twice_mod);
    vr3 = _mm512_hexl_small_mod_epu64<4>(vr3, v_modulus, &v_twice_mod);
    vr4 = _mm512_hexl_small_mod_epu64<4>(vr4, v_modulus, &v_twice_mod);
    vr5 = _mm512_hexl_small_mod_epu64<4>(vr5, v_modulus, &v_twice_mod);
    vr6 = _mm512_hexl_small_mod_epu64<4>(vr6, v_modulus, &v_twice_mod);
    vr7 = _mm512_hexl_small_mod_epu64<4>(vr7, v_modulus, &v_twice_mod);
    vr8 = _mm512_hexl_small_mod_epu64<4>(vr8, v_modulus, &v_twice_mod);
    vr9 = _mm512_hexl_small_mod_epu64<4>(vr9, v_modulus, &v_twice_mod);
    vr10 = _mm512_hexl_small_mod_epu64<4>(vr10, v_modulus, &v_twice_mod);
    vr11 = _mm512_hexl_small_mod_epu64<4>(vr11, v_modulus, &v_twice_mod);
    vr12 = _mm512_hexl_small_mod_epu64<4>(vr12, v_modulus, &v_twice_mod);
    vr13 = _mm512_hexl_small_mod_epu64<4>(vr13, v_modulus, &v_twice_mod);
    vr14 = _mm512_hexl_small_mod_epu64<4>(vr14, v_modulus, &v_twice_mod);
    vr15 = _mm512_hexl_small_mod_epu64<4>(vr15, v_modulus, &v_twice_mod);
    vr16 = _mm512_hexl_small_mod_epu64<4>(vr16, v_modulus, &v_twice_mod);

    _mm512_storeu_si512(vp_result++, vr1);
    _mm512_storeu_si512(vp_result++, vr2);
    _mm512_storeu_si512(vp_result++, vr3);
    _mm512_storeu_si512(vp_result++, vr4);
    _mm512_storeu_si512(vp_result++, vr5);
    _mm512_storeu_si512(vp_result++, vr6);
    _mm512_storeu_si512(vp_result++, vr7);
    _mm512_storeu_si512(vp_result++, vr8);
    _mm512_storeu_si512(vp_result++, vr9);
    _mm512_storeu_si512(vp_result++, vr10);
    _mm512_storeu_si512(vp_result++, vr11);
    _mm512_storeu_si512(vp_result++, vr12);
    _mm512_storeu_si512(vp_result++, vr13);
    _mm512_storeu_si512(vp_result++, vr14);
    _mm512_storeu_si512(vp_result++, vr15);
    _mm512_storeu_si512(vp_result++, vr16);
  }
}

// Algorithm 1 from
// https://hal.archives-ouvertes.fr/hal-01215845/document
template <int BitShift, int InputModFactor>
void EltwiseMultModAVX512IntLoopDefault(__m512i* vp_result,
                                        const __m512i* vp_operand1,
                                        const __m512i* vp_operand2,
                                        __m512i vbarr_lo, __m512i v_modulus,
                                        __m512i v_twice_mod, uint64_t n) {
  HEXL_UNUSED(v_twice_mod);
  HEXL_LOOP_UNROLL_4
  for (size_t i = n / 8; i > 0; --i) {
    __m512i v_operand1 = _mm512_loadu_si512(vp_operand1);
    __m512i v_operand2 = _mm512_loadu_si512(vp_operand2);

    v_operand1 = _mm512_hexl_small_mod_epu64<InputModFactor>(
        v_operand1, v_modulus, &v_twice_mod);

    v_operand2 = _mm512_hexl_small_mod_epu64<InputModFactor>(
        v_operand2, v_modulus, &v_twice_mod);

    __m512i vprod_hi = _mm512_hexl_mulhi_epi<64>(v_operand1, v_operand2);
    __m512i vprod_lo = _mm512_hexl_mullo_epi<64>(v_operand1, v_operand2);

    __m512i c1 = _mm512_hexl_shrdi_epi64<BitShift - 1>(vprod_lo, vprod_hi);
    __m512i c3 = _mm512_hexl_mulhi_approx_epi<64>(c1, vbarr_lo);
    __m512i vresult = _mm512_hexl_mullo_epi<64>(c3, v_modulus);
    // Computes result in [0, 4q)
    vresult = _mm512_sub_epi64(vprod_lo, vresult);

    // Reduce result to [0, q)
    vresult = _mm512_hexl_small_mod_epu64<4>(vresult, v_modulus, &v_twice_mod);
    _mm512_storeu_si512(vp_result, vresult);

    ++vp_operand1;
    ++vp_operand2;
    ++vp_result;
  }
}

template <int BitShift, int InputModFactor>
void EltwiseMultModAVX512IntLoop(__m512i* vp_result, const __m512i* vp_operand1,
                                 const __m512i* vp_operand2, __m512i vbarr_lo,
                                 __m512i v_modulus, __m512i v_twice_mod,
                                 uint64_t n) {
  switch (n) {
    case 1024:
      EltwiseMultModAVX512IntLoopUnroll<BitShift, InputModFactor, 1024>(
          vp_result, vp_operand1, vp_operand2, vbarr_lo, v_modulus,
          v_twice_mod);
      break;

    case 2048:
      EltwiseMultModAVX512IntLoopUnroll<BitShift, InputModFactor, 2048>(
          vp_result, vp_operand1, vp_operand2, vbarr_lo, v_modulus,
          v_twice_mod);
      break;

    case 4096:
      EltwiseMultModAVX512IntLoopUnroll<BitShift, InputModFactor, 4096>(
          vp_result, vp_operand1, vp_operand2, vbarr_lo, v_modulus,
          v_twice_mod);
      break;

    case 8192:
      EltwiseMultModAVX512IntLoopUnroll<BitShift, InputModFactor, 8192>(
          vp_result, vp_operand1, vp_operand2, vbarr_lo, v_modulus,
          v_twice_mod);
      break;

    case 16384:
      EltwiseMultModAVX512IntLoopUnroll<BitShift, InputModFactor, 16384>(
          vp_result, vp_operand1, vp_operand2, vbarr_lo, v_modulus,
          v_twice_mod);
      break;

    case 32768:
      EltwiseMultModAVX512IntLoopUnroll<BitShift, InputModFactor, 32768>(
          vp_result, vp_operand1, vp_operand2, vbarr_lo, v_modulus,
          v_twice_mod);
      break;

    default:
      EltwiseMultModAVX512IntLoopDefault<BitShift, InputModFactor>(
          vp_result, vp_operand1, vp_operand2, vbarr_lo, v_modulus, v_twice_mod,
          n);
  }
}

// Algorithm 1 from https://hal.archives-ouvertes.fr/hal-01215845/document
template <int InputModFactor>
void EltwiseMultModAVX512Int(uint64_t* result, const uint64_t* operand1,
                             const uint64_t* operand2, uint64_t n,
                             uint64_t modulus) {
  HEXL_CHECK(InputModFactor == 1 || InputModFactor == 2 || InputModFactor == 4,
             "Require InputModFactor = 1, 2, or 4")
  HEXL_CHECK(InputModFactor * modulus > (1ULL << 50),
             "Require InputModFactor * modulus > (1ULL << 50)")
  HEXL_CHECK(InputModFactor * modulus < (1ULL << 63),
             "Require InputModFactor * modulus < (1ULL << 63)");
  HEXL_CHECK(modulus < (1ULL << 62), "Require  modulus < (1ULL << 62)");
  HEXL_CHECK_BOUNDS(operand1, n, InputModFactor * modulus,
                    "operand1 exceeds bound " << (InputModFactor * modulus));
  HEXL_CHECK_BOUNDS(operand2, n, InputModFactor * modulus,
                    "operand2 exceeds bound " << (InputModFactor * modulus));
  HEXL_CHECK(modulus > 1, "Require modulus > 1");
  uint64_t n_mod_8 = n % 8;
  if (n_mod_8 != 0) {
    EltwiseMultModNative<InputModFactor>(result, operand1, operand2, n_mod_8,
                                         modulus);
    operand1 += n_mod_8;
    operand2 += n_mod_8;
    result += n_mod_8;
    n -= n_mod_8;
  }

  const uint64_t logmod = MSB(modulus);
  uint64_t log2_input_mod_factor = 0;
  if (InputModFactor == 2) {
    log2_input_mod_factor = 1;
  } else if (InputModFactor == 4) {
    log2_input_mod_factor = 2;
  }

  // modulus < 2**N
  const uint64_t N = logmod + 1;
  uint64_t L = 63 + N;  // Ensures L-N+1 == 64
  uint64_t op_hi = uint64_t(1) << (L - 64);
  uint64_t op_lo = uint64_t(0);
  uint64_t barr_lo = DivideUInt128UInt64Lo(op_hi, op_lo, modulus);

  // Let d be the product operand1 * operand2.
  // To ensure d >> (N - 1) < (1ULL << 64), we need
  // (input_mod_factor * modulus)^2 >> (N-1) < (1ULL << 64)
  // This happens when 2 * log_2(input_mod_factor) + N < 63
  // If not, we need to reduce the inputs to be less than modulus for
  // correctness. This is less efficient, so we avoid it when possible.
  bool reduce_mod = 2 * log2_input_mod_factor + N >= 63;

  __m512i vbarr_lo = _mm512_set1_epi64(static_cast<int64_t>(barr_lo));
  __m512i v_modulus = _mm512_set1_epi64(static_cast<int64_t>(modulus));
  __m512i v_twice_mod = _mm512_set1_epi64(static_cast<int64_t>(2 * modulus));
  const __m512i* vp_operand1 = reinterpret_cast<const __m512i*>(operand1);
  const __m512i* vp_operand2 = reinterpret_cast<const __m512i*>(operand2);
  __m512i* vp_result = reinterpret_cast<__m512i*>(result);

  if (reduce_mod) {
    // This case happens only when  N >= 63 - 2 * log2(input_mod_factor) = 59
    // Additionally, modulus < (1ULL << 62) implies N <= 62.
    // So N == 59, 60, 61, 62 are the only cases here.
    switch (N) {
      case 59: {
        EltwiseMultModAVX512IntLoop<59, InputModFactor>(
            vp_result, vp_operand1, vp_operand2, vbarr_lo, v_modulus,
            v_twice_mod, n);
        break;
      }
      case 60: {
        EltwiseMultModAVX512IntLoop<60, InputModFactor>(
            vp_result, vp_operand1, vp_operand2, vbarr_lo, v_modulus,
            v_twice_mod, n);
        break;
      }
      case 61: {
        EltwiseMultModAVX512IntLoop<61, InputModFactor>(
            vp_result, vp_operand1, vp_operand2, vbarr_lo, v_modulus,
            v_twice_mod, n);
        break;
      }
      case 62: {
        EltwiseMultModAVX512IntLoop<62, InputModFactor>(
            vp_result, vp_operand1, vp_operand2, vbarr_lo, v_modulus,
            v_twice_mod, n);
        break;
      }
      default: {
        HEXL_CHECK(false, "Bad value for N: " << N);
      }
    }
  } else {  // Input mod reduction not required; pass InputModFactor == 1.
    // For N < 50, we should prefer EltwiseMultModAVX512Float, so we don't
    // generate a special case for it here

    // The template arguments are required for use of _mm512_hexl_shrdi_epi64,
    // which requires a compile-time constant for the shift.
    switch (N) {
      case 50: {
        EltwiseMultModAVX512IntLoop<50, 1>(vp_result, vp_operand1, vp_operand2,
                                           vbarr_lo, v_modulus, v_twice_mod, n);
        break;
      }
      case 51: {
        EltwiseMultModAVX512IntLoop<51, 1>(vp_result, vp_operand1, vp_operand2,
                                           vbarr_lo, v_modulus, v_twice_mod, n);
        break;
      }
      case 52: {
        EltwiseMultModAVX512IntLoop<52, 1>(vp_result, vp_operand1, vp_operand2,
                                           vbarr_lo, v_modulus, v_twice_mod, n);
        break;
      }
      case 53: {
        EltwiseMultModAVX512IntLoop<53, 1>(vp_result, vp_operand1, vp_operand2,
                                           vbarr_lo, v_modulus, v_twice_mod, n);
        break;
      }
      case 54: {
        EltwiseMultModAVX512IntLoop<54, 1>(vp_result, vp_operand1, vp_operand2,
                                           vbarr_lo, v_modulus, v_twice_mod, n);
        break;
      }
      case 55: {
        EltwiseMultModAVX512IntLoop<55, 1>(vp_result, vp_operand1, vp_operand2,
                                           vbarr_lo, v_modulus, v_twice_mod, n);
        break;
      }
      case 56: {
        EltwiseMultModAVX512IntLoop<56, 1>(vp_result, vp_operand1, vp_operand2,
                                           vbarr_lo, v_modulus, v_twice_mod, n);
        break;
      }
      case 57: {
        EltwiseMultModAVX512IntLoop<57, 1>(vp_result, vp_operand1, vp_operand2,
                                           vbarr_lo, v_modulus, v_twice_mod, n);
        break;
      }
      case 58: {
        EltwiseMultModAVX512IntLoop<58, 1>(vp_result, vp_operand1, vp_operand2,
                                           vbarr_lo, v_modulus, v_twice_mod, n);
        break;
      }
      case 59: {
        EltwiseMultModAVX512IntLoop<59, 1>(vp_result, vp_operand1, vp_operand2,
                                           vbarr_lo, v_modulus, v_twice_mod, n);
        break;
      }
      case 60: {
        EltwiseMultModAVX512IntLoop<60, 1>(vp_result, vp_operand1, vp_operand2,
                                           vbarr_lo, v_modulus, v_twice_mod, n);
        break;
      }
      case 61: {
        EltwiseMultModAVX512IntLoop<61, 1>(vp_result, vp_operand1, vp_operand2,
                                           vbarr_lo, v_modulus, v_twice_mod, n);
        break;
      }
      default: {
        // Algorithm 1 from
        // https://hal.archives-ouvertes.fr/hal-01215845/document
        HEXL_LOOP_UNROLL_4
        for (size_t i = n / 8; i > 0; --i) {
          __m512i v_operand1 = _mm512_loadu_si512(vp_operand1);
          __m512i v_operand2 = _mm512_loadu_si512(vp_operand2);

          // Compute product
          __m512i vprod_hi = _mm512_hexl_mulhi_epi<64>(v_operand1, v_operand2);
          __m512i vprod_lo = _mm512_hexl_mullo_epi<64>(v_operand1, v_operand2);

          __m512i c1 = _mm512_hexl_shrdi_epi64(
              vprod_lo, vprod_hi, static_cast<unsigned int>(N - 1));

          // L - N + 1 == 64, so we only need high 64 bits
          // Perform approximate computation of high bits, as described on page
          // 7 of https://arxiv.org/pdf/2003.04510.pdf
          __m512i c3 = _mm512_hexl_mulhi_approx_epi<64>(c1, vbarr_lo);

          // C4 = prod_lo - (p * c3)_lo
          __m512i vresult = _mm512_hexl_mullo_epi<64>(c3, v_modulus);
          // Computes result in [0, 4q)
          vresult = _mm512_sub_epi64(vprod_lo, vresult);

          // Reduce result to [0, q)
          vresult =
              _mm512_hexl_small_mod_epu64<4>(vresult, v_modulus, &v_twice_mod);
          _mm512_storeu_si512(vp_result, vresult);

          HEXL_CHECK_BOUNDS((uint64_t*)(vp_result), 8, modulus,
                            "result exceeds bound");

          ++vp_operand1;
          ++vp_operand2;
          ++vp_result;
        }
      }
    }
  }
  HEXL_CHECK_BOUNDS(result, n, modulus, "result exceeds bound " << modulus);
}

// From Function 18, page 19 of https://arxiv.org/pdf/1407.3383.pdf
// See also Algorithm 2/3 of
// https://hal.archives-ouvertes.fr/hal-02552673/document
template <int InputModFactor>
inline void EltwiseMultModAVX512FloatLoopDefault(
    __m512i* vp_result, const __m512i* vp_operand1, const __m512i* vp_operand2,
    __m512d u, __m512d p, __m512i v_modulus, __m512i v_twice_mod, uint64_t n) {
  HEXL_UNUSED(v_twice_mod);

  constexpr int round_mode = (_MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);

  HEXL_LOOP_UNROLL_4
  for (size_t i = n / 8; i > 0; --i) {
    __m512i v_operand1 = _mm512_loadu_si512(vp_operand1);
    v_operand1 = _mm512_hexl_small_mod_epu64<InputModFactor>(
        v_operand1, v_modulus, &v_twice_mod);

    __m512i v_operand2 = _mm512_loadu_si512(vp_operand2);
    v_operand2 = _mm512_hexl_small_mod_epu64<InputModFactor>(
        v_operand2, v_modulus, &v_twice_mod);

    __m512d x = _mm512_cvt_roundepu64_pd(v_operand1, round_mode);
    __m512d y = _mm512_cvt_roundepu64_pd(v_operand2, round_mode);

    __m512d h = _mm512_mul_pd(x, y);
    __m512d l = _mm512_fmsub_pd(x, y, h);  // rounding error; h + l == x * y
    __m512d b = _mm512_mul_pd(h, u);       // ~ (x * y) / p
    __m512d c = _mm512_floor_pd(b);        // ~ floor(x * y / p)
    __m512d d = _mm512_fnmadd_pd(c, p, h);
    __m512d g = _mm512_add_pd(d, l);
    __mmask8 m = _mm512_cmp_pd_mask(g, _mm512_setzero_pd(), _CMP_LT_OQ);
    g = _mm512_mask_add_pd(g, m, g, p);

    __m512i v_result = _mm512_cvt_roundpd_epu64(g, round_mode);

    _mm512_storeu_si512(vp_result, v_result);

    ++vp_operand1;
    ++vp_operand2;
    ++vp_result;
  }
}

template <int InputModFactor, int CoeffCount>
inline void EltwiseMultModAVX512FloatLoopUnroll(
    __m512i* vp_result, const __m512i* vp_operand1, const __m512i* vp_operand2,
    __m512d u, __m512d p, __m512i v_modulus, __m512i v_twice_mod) {
  constexpr size_t manual_unroll_factor = 4;
  constexpr size_t avx512_64bit_count = 8;
  constexpr size_t loop_count =
      CoeffCount / (manual_unroll_factor * avx512_64bit_count);

  static_assert(loop_count > 0, "loop_count too small for unrolling");
  static_assert(CoeffCount % (manual_unroll_factor * avx512_64bit_count) == 0,
                "CoeffCount must be a factor of manual_unroll_factor * "
                "avx512_64bit_count");

  constexpr int round_mode = (_MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);

  HEXL_LOOP_UNROLL_4
  for (size_t i = loop_count; i > 0; --i) {
    __m512i op1_1 = _mm512_loadu_si512(vp_operand1++);
    __m512i op1_2 = _mm512_loadu_si512(vp_operand1++);
    __m512i op1_3 = _mm512_loadu_si512(vp_operand1++);
    __m512i op1_4 = _mm512_loadu_si512(vp_operand1++);

    __m512i op2_1 = _mm512_loadu_si512(vp_operand2++);
    __m512i op2_2 = _mm512_loadu_si512(vp_operand2++);
    __m512i op2_3 = _mm512_loadu_si512(vp_operand2++);
    __m512i op2_4 = _mm512_loadu_si512(vp_operand2++);

    op1_1 = _mm512_hexl_small_mod_epu64<InputModFactor>(op1_1, v_modulus,
                                                        &v_twice_mod);
    op1_2 = _mm512_hexl_small_mod_epu64<InputModFactor>(op1_2, v_modulus,
                                                        &v_twice_mod);
    op1_3 = _mm512_hexl_small_mod_epu64<InputModFactor>(op1_3, v_modulus,
                                                        &v_twice_mod);
    op1_4 = _mm512_hexl_small_mod_epu64<InputModFactor>(op1_4, v_modulus,
                                                        &v_twice_mod);

    op2_1 = _mm512_hexl_small_mod_epu64<InputModFactor>(op2_1, v_modulus,
                                                        &v_twice_mod);
    op2_2 = _mm512_hexl_small_mod_epu64<InputModFactor>(op2_2, v_modulus,
                                                        &v_twice_mod);
    op2_3 = _mm512_hexl_small_mod_epu64<InputModFactor>(op2_3, v_modulus,
                                                        &v_twice_mod);
    op2_4 = _mm512_hexl_small_mod_epu64<InputModFactor>(op2_4, v_modulus,
                                                        &v_twice_mod);

    __m512d x1 = _mm512_cvt_roundepu64_pd(op1_1, round_mode);
    __m512d x2 = _mm512_cvt_roundepu64_pd(op1_2, round_mode);
    __m512d x3 = _mm512_cvt_roundepu64_pd(op1_3, round_mode);
    __m512d x4 = _mm512_cvt_roundepu64_pd(op1_4, round_mode);

    __m512d y1 = _mm512_cvt_roundepu64_pd(op2_1, round_mode);
    __m512d y2 = _mm512_cvt_roundepu64_pd(op2_2, round_mode);
    __m512d y3 = _mm512_cvt_roundepu64_pd(op2_3, round_mode);
    __m512d y4 = _mm512_cvt_roundepu64_pd(op2_4, round_mode);

    __m512d h1 = _mm512_mul_pd(x1, y1);
    __m512d h2 = _mm512_mul_pd(x2, y2);
    __m512d h3 = _mm512_mul_pd(x3, y3);
    __m512d h4 = _mm512_mul_pd(x4, y4);

    // ~ (x * y) / p
    __m512d b1 = _mm512_mul_pd(h1, u);
    __m512d b2 = _mm512_mul_pd(h2, u);
    __m512d b3 = _mm512_mul_pd(h3, u);
    __m512d b4 = _mm512_mul_pd(h4, u);

    // rounding error; h + l == x * y
    __m512d l1 = _mm512_fmsub_pd(x1, y1, h1);
    __m512d l2 = _mm512_fmsub_pd(x2, y2, h2);
    __m512d l3 = _mm512_fmsub_pd(x3, y3, h3);
    __m512d l4 = _mm512_fmsub_pd(x4, y4, h4);

    // ~ floor(x * y / p)
    __m512d c1 = _mm512_floor_pd(b1);
    __m512d c2 = _mm512_floor_pd(b2);
    __m512d c3 = _mm512_floor_pd(b3);
    __m512d c4 = _mm512_floor_pd(b4);

    __m512d d1 = _mm512_fnmadd_pd(c1, p, h1);
    __m512d d2 = _mm512_fnmadd_pd(c2, p, h2);
    __m512d d3 = _mm512_fnmadd_pd(c3, p, h3);
    __m512d d4 = _mm512_fnmadd_pd(c4, p, h4);

    __m512d g1 = _mm512_add_pd(d1, l1);
    __m512d g2 = _mm512_add_pd(d2, l2);
    __m512d g3 = _mm512_add_pd(d3, l3);
    __m512d g4 = _mm512_add_pd(d4, l4);

    __mmask8 m1 = _mm512_cmp_pd_mask(g1, _mm512_setzero_pd(), _CMP_LT_OQ);
    __mmask8 m2 = _mm512_cmp_pd_mask(g2, _mm512_setzero_pd(), _CMP_LT_OQ);
    __mmask8 m3 = _mm512_cmp_pd_mask(g3, _mm512_setzero_pd(), _CMP_LT_OQ);
    __mmask8 m4 = _mm512_cmp_pd_mask(g4, _mm512_setzero_pd(), _CMP_LT_OQ);

    g1 = _mm512_mask_add_pd(g1, m1, g1, p);
    g2 = _mm512_mask_add_pd(g2, m2, g2, p);
    g3 = _mm512_mask_add_pd(g3, m3, g3, p);
    g4 = _mm512_mask_add_pd(g4, m4, g4, p);

    __m512i out1 = _mm512_cvt_roundpd_epu64(g1, round_mode);
    __m512i out2 = _mm512_cvt_roundpd_epu64(g2, round_mode);
    __m512i out3 = _mm512_cvt_roundpd_epu64(g3, round_mode);
    __m512i out4 = _mm512_cvt_roundpd_epu64(g4, round_mode);

    _mm512_storeu_si512(vp_result++, out1);
    _mm512_storeu_si512(vp_result++, out2);
    _mm512_storeu_si512(vp_result++, out3);
    _mm512_storeu_si512(vp_result++, out4);
  }
}

template <int InputModFactor>
inline void EltwiseMultModAVX512FloatLoop(__m512i* vp_result,
                                          const __m512i* vp_operand1,
                                          const __m512i* vp_operand2, __m512d u,
                                          __m512d p, __m512i v_modulus,
                                          __m512i v_twice_mod, uint64_t n) {
  switch (n) {
    case 1024:
      EltwiseMultModAVX512FloatLoopUnroll<InputModFactor, 1024>(
          vp_result, vp_operand1, vp_operand2, u, p, v_modulus, v_twice_mod);
      break;

    case 2048:
      EltwiseMultModAVX512FloatLoopUnroll<InputModFactor, 2048>(
          vp_result, vp_operand1, vp_operand2, u, p, v_modulus, v_twice_mod);
      break;

    case 4096:
      EltwiseMultModAVX512FloatLoopUnroll<InputModFactor, 4096>(
          vp_result, vp_operand1, vp_operand2, u, p, v_modulus, v_twice_mod);
      break;

    case 8192:
      EltwiseMultModAVX512FloatLoopUnroll<InputModFactor, 8192>(
          vp_result, vp_operand1, vp_operand2, u, p, v_modulus, v_twice_mod);
      break;

    case 16384:
      EltwiseMultModAVX512FloatLoopUnroll<InputModFactor, 16384>(
          vp_result, vp_operand1, vp_operand2, u, p, v_modulus, v_twice_mod);
      break;

    case 32768:
      EltwiseMultModAVX512FloatLoopUnroll<InputModFactor, 32768>(
          vp_result, vp_operand1, vp_operand2, u, p, v_modulus, v_twice_mod);
      break;

    default:
      EltwiseMultModAVX512FloatLoopDefault<InputModFactor>(
          vp_result, vp_operand1, vp_operand2, u, p, v_modulus, v_twice_mod, n);
  }
}

// From Function 18, page 19 of https://arxiv.org/pdf/1407.3383.pdf
// See also Algorithm 2/3 of
// https://hal.archives-ouvertes.fr/hal-02552673/document
template <int InputModFactor>
void EltwiseMultModAVX512Float(uint64_t* result, const uint64_t* operand1,
                               const uint64_t* operand2, uint64_t n,
                               uint64_t modulus) {
  HEXL_CHECK(modulus < MaximumValue(50),
             " modulus " << modulus << " exceeds bound " << MaximumValue(50));
  HEXL_CHECK(modulus > 1, "Require modulus > 1");

  HEXL_CHECK_BOUNDS(operand1, n, InputModFactor * modulus,
                    "operand1 exceeds bound " << (InputModFactor * modulus));
  HEXL_CHECK_BOUNDS(operand2, n, InputModFactor * modulus,
                    "operand2 exceeds bound " << (InputModFactor * modulus));
  uint64_t n_mod_8 = n % 8;
  if (n_mod_8 != 0) {
    EltwiseMultModNative<InputModFactor>(result, operand1, operand2, n_mod_8,
                                         modulus);
    operand1 += n_mod_8;
    operand2 += n_mod_8;
    result += n_mod_8;
    n -= n_mod_8;
  }
  __m512d p = _mm512_set1_pd(static_cast<double>(modulus));
  __m512i v_modulus = _mm512_set1_epi64(static_cast<int64_t>(modulus));
  __m512i v_twice_mod = _mm512_set1_epi64(static_cast<int64_t>(modulus * 2));

  // Add epsilon to ensure u * p >= 1.0
  // See Proposition 13 of https://arxiv.org/pdf/1407.3383.pdf
  double ubar = (1.0 + std::numeric_limits<double>::epsilon()) /
                static_cast<double>(modulus);
  __m512d u = _mm512_set1_pd(ubar);

  const __m512i* vp_operand1 = reinterpret_cast<const __m512i*>(operand1);
  const __m512i* vp_operand2 = reinterpret_cast<const __m512i*>(operand2);
  __m512i* vp_result = reinterpret_cast<__m512i*>(result);

  bool no_reduce_mod = (InputModFactor * modulus) < MaximumValue(50);
  if (no_reduce_mod) {  // No input modulus reduction necessary
    EltwiseMultModAVX512FloatLoop<1>(vp_result, vp_operand1, vp_operand2, u, p,
                                     v_modulus, v_twice_mod, n);
  } else {
    EltwiseMultModAVX512FloatLoop<InputModFactor>(
        vp_result, vp_operand1, vp_operand2, u, p, v_modulus, v_twice_mod, n);
  }

  HEXL_CHECK_BOUNDS(result, n, modulus, "result exceeds bound " << modulus);
}

#endif  // HEXL_HAS_AVX512DQ

}  // namespace hexl
}  // namespace intel
