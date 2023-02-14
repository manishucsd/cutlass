#include "cutlass/aligned_buffer.h"
#include "cutlass/array.h"
#include "cutlass/core_io.h"
#include "cutlass/cutlass.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/threadblock/default_mma.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm80.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/transform/threadblock/predicated_tile_access_iterator.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/tensor_view_io.h"


////////////////////////////////////////////////////////////////////////////////
///          Typenames for CUTLASS Threadblock-level matmul. 
///          Ideally these are nested in templates, but we 
///          are not using templates here and just hardcoding 
///          the types for simplicity of the example)
////////////////////////////////////////////////////////////////////////////////  
using ElementA = float;
using LayoutA = cutlass::layout::RowMajor;
using ElementB = float;
using LayoutB = cutlass::layout::RowMajor;
using ElementC = float;
using LayoutC = cutlass::layout::RowMajor;
using ElementAccumulator = float;

// Derived types for IREE matmul
using ElementLhs = ElementA;
using ElementRhs = ElementB;
using ElementResult = ElementC;

using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;

using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>;

static int const kStages = 3;
static int const kAlignmentA = 4;
static int const kAlignmentB = 4;

// CUTLASS Threadblock-level matmul operator and globlal memory tile iterators
using DefaultMma = typename cutlass::gemm::threadblock::DefaultMma<
    ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB,
    ElementAccumulator, LayoutC, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    ThreadblockShape, WarpShape, InstructionShape, kStages,  cutlass::arch::OpMultiplyAdd>;

// CUTLASS Threadblock-level multistage matrix multiply-accumulate pipeline
using ThreadblockMma = typename DefaultMma::ThreadblockMma;
using IteratorA = typename ThreadblockMma::IteratorA;
using IteratorB = typename ThreadblockMma::IteratorB;

////////////////////////////////////////////////////////////////////////////////
///          Naive Threadblock-level matmul 
////////////////////////////////////////////////////////////////////////////////
__device__ void iree_naive_microkernel(
  ElementLhs *lhs, ElementRhs *rhs, ElementResult *res, 
  cutlass::gemm::GemmCoord problem_size,
  int m_tile_offset, // Tile offset for M dimension in number of elements
  int n_tile_offset  // Tile offset for N dimension in number of elements
  ) {
  
  static int const kTileM = ThreadblockShape::kM;
  static int const kTileN = ThreadblockShape::kN;

  // Move pointers to the start of the tile for row-major MxK matrixA (LHS)
  // Move pointers to the start of the tile for row-major KxN matrixB (RHS)
  // Move pointers to the start of the tile for row-major MxN matrixC (Result)
  ElementLhs     *p_lhs = &lhs[m_tile_offset * problem_size.k()];
  ElementRhs     *p_rhs = &rhs[n_tile_offset];
  ElementResult  *p_res = &res[m_tile_offset * problem_size.n() + n_tile_offset];


  // Distributed the work across the thread.x and thread.y within a threadblock of (blockDim.x, blockDim.y)
  for(int m_thread_offset = threadIdx.x; m_thread_offset < kTileM; m_thread_offset += blockDim.x) {
    for(int n_thread_offset = threadIdx.y; n_thread_offset <  kTileN; n_thread_offset += blockDim.y) {
      // Accumulate the result in the k dimension
      int res_idx = m_thread_offset * problem_size.n() + n_thread_offset;
      p_res[res_idx] = 0;
      for (int k_offset = 0; k_offset < problem_size.k(); k_offset++) {
        int lhs_idx = m_thread_offset * problem_size.k() + k_offset;
        int rhs_idx = k_offset * problem_size.n() + n_thread_offset;
        p_res[res_idx] += p_lhs[lhs_idx] * p_rhs[rhs_idx];
      }
    }
  }
}


////////////////////////////////////////////////////////////////////////////////
///          CUTLASS Threadblock-level matmul 
////////////////////////////////////////////////////////////////////////////////
__device__ void iree_cutlass_microkernel(
  ElementLhs *lhs, ElementRhs *rhs, ElementResult *res, 
  cutlass::gemm::GemmCoord problem_size,
  int m_tile_offset, // Tile offset for M dimension in number of elements
  int n_tile_offset  // Tile offset for N dimension in number of elements
  ) {

  // Dynamic shared memory base pointer
  extern __shared__ int GemmSharedStorageBase[];

  // Declare pointer to dynamic shared memory.
  typename ThreadblockMma::SharedStorage *shared_storage =
      reinterpret_cast<typename ThreadblockMma::SharedStorage *>(GemmSharedStorageBase);

  // Compute threadblock location
  cutlass::gemm::GemmCoord tb_tile_offset = {int(blockIdx.x), int(blockIdx.y), 0};

  cutlass::MatrixCoord tb_offset_A{tb_tile_offset.m() * ThreadblockMma::Shape::kM,
                                   tb_tile_offset.k()};

  cutlass::MatrixCoord tb_offset_B{tb_tile_offset.k(),
                                   tb_tile_offset.n() * ThreadblockMma::Shape::kN};

  // Compute position within threadblock (linearized thread ID)
  int tb_thread_id = threadIdx.y * blockDim.x + threadIdx.x;
  int lane_id = tb_thread_id & 0x1f;

  typename IteratorA::Params params_A(cutlass::layout::RowMajor::packed({problem_size.m(), problem_size.k()}));
  typename IteratorB::Params params_B(cutlass::layout::RowMajor::packed({problem_size.k(), problem_size.n()}));

  // Construct iterators to A and B operands
  typename ThreadblockMma::IteratorA iterator_A(params_A, lhs,
                                     {problem_size.m(), problem_size.k()},
                                     tb_thread_id, tb_offset_A);

  typename ThreadblockMma::IteratorB iterator_B(params_B, rhs,
                                     {problem_size.k(), problem_size.n()},
                                     tb_thread_id, tb_offset_B);

  int warp_id = __shfl_sync(0xffffffff, threadIdx.y, 0);

  // Construct thread-scoped matrix multiply
  ThreadblockMma mma(*shared_storage, tb_thread_id, warp_id, lane_id);

  typename ThreadblockMma::FragmentC accum;

  accum.clear();

  int gemm_k_iterations = (problem_size.k() + ThreadblockMma::Shape::kK - 1) / ThreadblockMma::Shape::kK;

  // Compute threadblock-scoped matrix multiply-add
  mma(gemm_k_iterations, accum, iterator_A, iterator_B, accum);

  // Store accumulators to output tile
  typename ThreadblockMma::Operator::IteratorC iterator_C({res, problem_size.n()}, threadIdx.x);

  iterator_C.add_tile_offset(
      {(tb_tile_offset.m() * ThreadblockMma::WarpCount::kM) +
           (warp_id % ThreadblockMma::WarpCount::kM),
       (tb_tile_offset.n() * ThreadblockMma::WarpCount::kN) +
           (warp_id / ThreadblockMma::WarpCount::kM)});

  iterator_C.store(accum);
}

/// Device-level IREE kernel call
__global__ void iree_kernel(
  ElementLhs *lhs, ElementRhs *rhs, ElementResult *res,
  cutlass::gemm::GemmCoord problem_size ) {  
    
  static int const kTileM = ThreadblockShape::kM;
  static int const kTileN = ThreadblockShape::kN;

  // Distributed the work across the block.x and block.y within a grid/kernel of (gridDim.x, gridDim.y)
  for( int m_tile_offset = blockIdx.y * kTileM; m_tile_offset < problem_size.m(); m_tile_offset += (gridDim.y * kTileM)) {
    for( int n_tile_offset = blockIdx.x * kTileN; n_tile_offset < problem_size.n(); n_tile_offset+= (gridDim.x * kTileN)) {      

      iree_cutlass_microkernel(lhs, rhs, res, problem_size, m_tile_offset, n_tile_offset);

      // iree_naive_microkernel(lhs, rhs, res, problem_size, m_tile_offset, n_tile_offset);

    }
  }
}

/// Helper to initialize a tensor view
template <typename Element, typename Layout>
bool initialize_tensor(
  cutlass::TensorView<Element, Layout> view,
  cutlass::Distribution::Kind dist_kind,
  uint64_t seed) {

  if (dist_kind == cutlass::Distribution::Uniform) {

    double scope_max, scope_min;
    int bits_input = cutlass::sizeof_bits<Element>::value;

    if (bits_input == 1) {
      scope_max = 2;
      scope_min = 0;
    } else if (bits_input <= 8) {
      scope_max = 2;
      scope_min = -2;
    } else {
      scope_max = 8;
      scope_min = -8;
    }

    cutlass::reference::host::TensorFillRandomUniform(
      view, seed, scope_max, scope_min, 0);
  }
  else if (dist_kind == cutlass::Distribution::Identity) {

    cutlass::reference::host::TensorFillIdentity(view);
  }
  else if (dist_kind == cutlass::Distribution::Gaussian) {

    cutlass::reference::host::TensorFillRandomGaussian(view, seed, 0, 0.5);
  }
  else if (dist_kind == cutlass::Distribution::Sequential) {

    cutlass::reference::host::BlockFillSequential(
      view.data(), view.capacity());
  }
  else {
    // TODO: Implement the rest
    std::cerr << "Not implemented";
    return false;
  }

  return true;
}

int main() { 
  // CUTLASS Threadblock-level multistage matrix multiply-accumulate pipeline
  using ThreadblockMma = typename DefaultMma::ThreadblockMma;

  // Create a GEMM problem_size (m, n, k), alpha, beta
  cutlass::gemm::GemmCoord problem_size(128, 128, 64);
  float alpha = 1.0f;
  float beta = 0.0f;

  cutlass::HostTensor<ElementA, LayoutA> matrix_A;
  cutlass::HostTensor<ElementB, LayoutB> matrix_B;
  cutlass::HostTensor<ElementC, LayoutC> matrix_C_computed;
  cutlass::HostTensor<ElementC, LayoutC> matrix_C_reference;


  // Allocate device and host memory
  matrix_A.resize(problem_size.mk());
  matrix_B.resize(problem_size.kn());
  matrix_C_computed.resize(problem_size.mn());
  matrix_C_reference.resize(problem_size.mn(), false);

  //
  // initialize device memory
  //
  uint64_t seed = 2080;
  cutlass::Distribution::Kind init_A = cutlass::Distribution::Uniform;
  cutlass::Distribution::Kind init_B = cutlass::Distribution::Uniform;
  cutlass::Distribution::Kind init_C = cutlass::Distribution::Uniform;

  initialize_tensor(matrix_A.host_view(), init_A, seed + 2019);
  initialize_tensor(matrix_B.host_view(), init_B, seed + 2018);
  initialize_tensor(matrix_C_computed.host_view(), init_C, seed + 2017);
  cutlass::reference::host::TensorCopy(matrix_C_reference.host_view(), matrix_C_computed.host_view());

  // Sync device memory (copy host to device)
  matrix_A.sync_device();
  matrix_B.sync_device();
  matrix_C_computed.sync_device();

  cudaError_t result;

  // If requires more than 48KB: configure for extended, dynamic shared memory
  int smem_size = int(sizeof(typename ThreadblockMma::SharedStorage));
  std::cout << "smem_size: " << smem_size << std::endl;
  cudaDeviceProp properties;
  int device_idx;
  
  result = cudaGetDevice(&device_idx);
  if (result != cudaSuccess) {
    throw std::runtime_error("cudaGetDevice() API call failed.");
  }

  result = cudaGetDeviceProperties(&properties, device_idx);
  if (result != cudaSuccess) {
    throw std::runtime_error("cudaGetDeviceProperties() failed");
  }

  if (properties.sharedMemPerBlockOptin < smem_size) {
    std::cerr << "Shared memory size (" << properties.sharedMemPerBlockOptin << " bytes) "
              << "exceeds the device limit. Please use a device with more shared memory." 
              << std::endl;
    throw std::runtime_error("cudaGetDeviceProperties() failed");
  }

  if (smem_size >= (48 << 10)) {
    result = cudaFuncSetAttribute(iree_kernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, 
                                  smem_size);
    if (result != cudaSuccess) {
      std::cerr << "cudaFuncSetAttribute / cudaFuncAttributeMaxDynamicSharedMemorySize failed: " << cudaGetErrorString(result) << std::endl;
      return 1;
    }

    // Carveout 100% shared memory for this kernel.
    result = cudaFuncSetAttribute(iree_kernel,
                                  cudaFuncAttributePreferredSharedMemoryCarveout, 100);

    if (result != cudaSuccess) {
      std::cerr << "cudaFuncSetAttribute/ cudaFuncAttributePreferredSharedMemoryCarveout failed: " << cudaGetErrorString(result) << std::endl;
      return 1;
    }
  }

  // TODO: Host-constructible threadblock iterator params (micro kernel constructing on device, not optimal)
  // typename IteratorA::Params param_A(matrix_A.layout());
  // typename IteratorB::Params param_B(matrix_B.layout());

  // TODO: Get launch grid and block size procedurally 
  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  ElementLhs *lhs = matrix_A.device_data();
  ElementRhs *rhs = matrix_B.device_data();
  ElementResult *res = matrix_C_computed.device_data();
  
  iree_kernel<<<grid, block, smem_size>>>(lhs, rhs, res, problem_size);

  //
  // Check error code
  //

  result = cudaDeviceSynchronize();
  if (result != cudaSuccess) {
      std::cerr << " kernel error: " << cudaGetErrorString(result);
  }

  matrix_C_computed.sync_host();


  // VERFIY HERE
  cutlass::reference::host::Gemm<ElementA, LayoutA, ElementB, LayoutB, ElementC,
                                 LayoutC, ElementC, ElementC>reference_gemm;
  reference_gemm(problem_size, ElementC(alpha), matrix_A.host_view(),
                 matrix_B.host_view(), ElementC(beta),
                 matrix_C_reference.host_view());

  bool passed = cutlass::reference::host::TensorEquals(
      matrix_C_computed.host_view(), matrix_C_reference.host_view());

  if (!passed) {
    std::cout << __FILE__ << ":" << __LINE__ << "  "
              << "A:\n"
              << matrix_A.host_view() << "\n"
              << "B:\n"
              << matrix_B.host_view() << "\n"
              << "Reference:\n"
              << matrix_C_reference.host_view() << "\n"
              << "Computed:\n"
              << matrix_C_computed.host_view() << "\n";
  }

  printf("VERIFY: %s\n", passed ? "SUCCESS" : "FAIL");

  return 0;
}