/***************************************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

 /*! \file
    \brief Tests for device-wide GEMM interface
*/

#include <iostream>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "gemm_operation_3x.hpp"
#include "../common/cutlass_unit_test.h"


#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                          Create GEMM Device Operator
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
using cutlass3x_sm90_tensorop_s64x128x32gemm_e4m3_e4m3_f32_bf16_bf16_128x128x128_1x2x1_0_tnn_align16_warpspecialized_cooperative_epi_tma_epilogue =
  typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    cute::Shape<cute::_128, cute::_128, cute::_128>,
    cute::Shape<cute::_1, cute::_2, cute::_1>,
    cutlass::epilogue::collective::EpilogueTileAuto,
    float, float,
    cutlass::bfloat16_t, cutlass::layout::ColumnMajor, 1,
    cutlass::bfloat16_t, cutlass::layout::ColumnMajor, 1,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::epilogue::fusion::LinearCombination<
      cutlass::bfloat16_t,
      float,
      cutlass::bfloat16_t,
      float
    >
  >::CollectiveOp;

using cutlass3x_sm90_tensorop_s64x128x32gemm_e4m3_e4m3_f32_bf16_bf16_128x128x128_1x2x1_0_tnn_align16_warpspecialized_cooperative_epi_tma_mainloop =
  typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    cutlass::float_e4m3_t, cutlass::layout::RowMajor, 16,
    cutlass::float_e4m3_t, cutlass::layout::ColumnMajor, 16,
    float,
    cute::Shape<cute::_128, cute::_128, cute::_128>,
    cute::Shape<cute::_1, cute::_2, cute::_1>,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename cutlass3x_sm90_tensorop_s64x128x32gemm_e4m3_e4m3_f32_bf16_bf16_128x128x128_1x2x1_0_tnn_align16_warpspecialized_cooperative_epi_tma_epilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative
  >::CollectiveOp;

// Gemm operator cutlass3x_sm90_tensorop_s64x128x32gemm_e4m3_e4m3_f32_bf16_bf16_128x128x128_1x2x1_0_tnn_align16_warpspecialized_cooperative_epi_tma
using cutlass3x_sm90_tensorop_s64x128x32gemm_e4m3_e4m3_f32_bf16_bf16_128x128x128_1x2x1_0_tnn_align16_warpspecialized_cooperative_epi_tma_base = 
cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int,int,int,int>,
    cutlass3x_sm90_tensorop_s64x128x32gemm_e4m3_e4m3_f32_bf16_bf16_128x128x128_1x2x1_0_tnn_align16_warpspecialized_cooperative_epi_tma_mainloop,
    cutlass3x_sm90_tensorop_s64x128x32gemm_e4m3_e4m3_f32_bf16_bf16_128x128x128_1x2x1_0_tnn_align16_warpspecialized_cooperative_epi_tma_epilogue,
    void>;

// Create Gemm Device Operator
// Define named type
struct cutlass3x_sm90_tensorop_s64x128x32gemm_e4m3_e4m3_f32_bf16_bf16_128x128x128_1x2x1_0_tnn_align16_warpspecialized_cooperative_epi_tma :
  public cutlass3x_sm90_tensorop_s64x128x32gemm_e4m3_e4m3_f32_bf16_bf16_128x128x128_1x2x1_0_tnn_align16_warpspecialized_cooperative_epi_tma_base { };

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


TEST(sm90_gemm_description_s64x128x32gemm_e4m3_e4m3_f32_bf16_bf16_tnn, 128x128x128_1x2x1_0_align16_warpspecialized_cooperative_epi_tma_epilogue) {
  
  using GemmDeviceOperator = cutlass::gemm::device::GemmUniversalAdapter<
    cutlass3x_sm90_tensorop_s64x128x32gemm_e4m3_e4m3_f32_bf16_bf16_128x128x128_1x2x1_0_tnn_align16_warpspecialized_cooperative_epi_tma>;

  std::unique_ptr<cutlass::library::GemmUniversal3xOperation<GemmDeviceOperator>> op = 
    std::make_unique<cutlass::library::GemmUniversal3xOperation<GemmDeviceOperator>>(
    "cutlass3x_sm90_tensorop_s64x128x32gemm_e4m3_e4m3_f32_bf16_bf16_128x128x128_1x2x1_0_tnn_align16_warpspecialized_cooperative_epi_tma");


  std::cout << op.get()->description().name << std::endl;
  cutlass::library::GemmDescription const &gemm_desc = static_cast<cutlass::library::GemmDescription const&>(op.get()->description());
  
  //
  // Validate Funcational attribute (anything that can change the functionality of the kernel)
  //

  // Compile-time attributes read from the mainloop builder cutlass::gemm::collective::CollectiveBuilder
  EXPECT_TRUE(gemm_desc.A.element == cutlass::library::NumericTypeID::kFE4M3);
  EXPECT_TRUE(gemm_desc.A.layout == cutlass::library::LayoutTypeID::kRowMajor);
  EXPECT_TRUE(gemm_desc.B.element == cutlass::library::NumericTypeID::kFE4M3);
  EXPECT_TRUE(gemm_desc.B.layout == cutlass::library::LayoutTypeID::kColumnMajor);

  // Compile-time attributes read from the epilogue builder cutlass::epilogue::collective::CollectiveBuilder
  EXPECT_TRUE(gemm_desc.C.element == cutlass::library::NumericTypeID::kBF16);
  EXPECT_TRUE(gemm_desc.C.layout == cutlass::library::LayoutTypeID::kColumnMajor);
  EXPECT_TRUE(gemm_desc.D.element == cutlass::library::NumericTypeID::kBF16);
  EXPECT_TRUE(gemm_desc.D.layout == cutlass::library::LayoutTypeID::kColumnMajor);
  EXPECT_TRUE(gemm_desc.tile_description.math_instruction.element_accumulator == cutlass::library::NumericTypeID::kF32);
  EXPECT_TRUE(gemm_desc.element_epilogue == cutlass::library::NumericTypeID::kF32);

  //
  // Validate Performance attributes (anything that can vary the performance of the kernel with the same functionality)
  //

  // Compile-time attributes read from the mainloop builder cutlass::gemm::collective::CollectiveBuilder
  EXPECT_TRUE(gemm_desc.tile_description.threadblock_shape == cutlass::gemm::GemmCoord(128, 128, 128));
  EXPECT_TRUE(gemm_desc.tile_description.cluster_shape == cutlass::gemm::GemmCoord(1, 2, 1));
  // EXPECT_TRUE(gemm_desc.mainloop_kind == cutlass::library::MainloopKind::kWarpSpecializedCooperative); // TODO: Uncomment when the enum is added
  
  // Compile-time attributes read from the epilogue builder cutlass::epilogue::collective::CollectiveBuilder
  // EXPECT_TRUE(gemm_desc.epilogue_kind == cutlass::library::EpilogueKind::kTma);  // TODO: Uncomment when the enum is added
  
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                          Create GEMM Device Operator
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 

// Next Sm90 GEMM Operator test here

#endif // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
