# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name, no-value-for-parameter
"""Defines common C code for all microkernel operations."""


common_includes = """

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <rvp_intrinsic.h>

#include <tvm/runtime/crt/error_codes.h>

#ifndef READPAD
#define READPAD
static inline __attribute__((always_inline)) int32_t arm_nn_read_s8x4_ia(const int8_t **in_s8)
{
    int32_t val;
    memcpy(&val, *in_s8, 4);
    *in_s8 += 4;

    return (val);
}

static inline __attribute__((always_inline)) const int8_t *read_and_pad(const int8_t *source, int32_t *out1, int32_t *out2)
{
    int32_t inA = arm_nn_read_s8x4_ia(&source);
    int32_t inAbuf1 = __rv_sunpkd831((uint32_t)inA);
    int32_t inAbuf2 = __rv_sunpkd820(inA);

#ifndef ARM_MATH_BIG_ENDIAN
    *out2 = (int32_t)(__rv_pktb16(inAbuf1, inAbuf2));
    *out1 = (int32_t)(__rv_pkbt16(inAbuf2, inAbuf1));
#else
    *out1 = (int32_t)(__rv_pktb16(inAbuf1, inAbuf2));
    *out2 = (int32_t)(__rv_pkbt16(inAbuf2, inAbuf1));
#endif

    return source;
}
#endif 



"""

MICRO_WORD_LENGTH_BITS = 32


def num_simd_lanes_per_word(dtype: str) -> int:
    """Takes a dtype, and returns how many of that dtype fit into a single microcontroller word.

    >>> num_simd_lanes_per_word("int8")
    4
    >>> num_simd_lanes_per_word("int16")
    2
    """
    assert dtype.startswith("int")
    dtype_width = int(dtype[3:])
    return MICRO_WORD_LENGTH_BITS // dtype_width
