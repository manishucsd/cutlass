### Define CUTLASS_LIBRARY_KERNELS strings for different GEMM types
BF16="\
cutlass3x_sm100_tensorop_gemm_bf16_bf16_f32_*_tnt_*"

FP8="\
cutlass3x_sm100_tensorop_gemm_e4m3_e4m3_f32_*_tnt_*"

FE2M1="\
cutlass3x_sm100_bstensorop_gemm_ue4m3xe2m1_ue4m3xe2m1_f32_*_tnt_*"

GROUPED="\
cutlass3x_sm100_tensorop_gemm_grouped_bf16_bf16_f32_*_tnt_*"

#### Parse the command-line arguments
GEMM_TYPE="bf16"

# Loop through the command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -gemm-type|--gemm-type)
            GEMM_TYPE="$2"
            shift 2  # Skip the option and its value
            ;;
        *)
            echo "Unknown parameter passed: $1"
            shift
            ;;
    esac
done

### Set the CUTLASS_LIBRARY_KERNELS based on the GEMM_TYPE
case $GEMM_TYPE in
    bf16)
        CUTLASS_LIBRARY_KERNELS=$BF16
        ;;
    fp8)
        CUTLASS_LIBRARY_KERNELS=$FP8
        ;;
    fe1m2)
        CUTLASS_LIBRARY_KERNELS=$FE2M1
        ;;
    grouped)
        CUTLASS_LIBRARY_KERNELS=$GROUPED
        ;;
    all)
        CUTLASS_LIBRARY_KERNELS="$BF16,$FP8,$FE2M1,$GROUPED"
        ;;
    *)
        echo "Unknown GEMM_TYPE: $GEMM_TYPE"
        exit 1
        ;;
esac


### Check if sccache is available in the path
if command -v sccache &> /dev/null; then
    echo "sccache found, enabling compiler caching"
    COMPILER_CACHE_FLAGS+=" -DCMAKE_CXX_COMPILER_LAUNCHER=sccache"
    COMPILER_CACHE_FLAGS+=" -DCMAKE_CUDA_COMPILER_LAUNCHER=sccache"
    COMPILER_CACHE_FLAGS+=" -DCMAKE_C_COMPILER_LAUNCHER=sccache"
else
    echo "sccache not found, continuing without compiler caching"
fi

# Set default values for CMAKE/CUTLASS flags if not already set by the user
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
CUTLASS_CUDA_NVCC_FLAGS="${CUTLASS_CUDA_NVCC_FLAGS:-}"
### Run the cmake command
set -x

cmake \
-DCMAKE_BUILD_TYPE:STRING="$CMAKE_BUILD_TYPE" \
-DCUTLASS_CUDA_NVCC_FLAGS:STRING="$CUTLASS_CUDA_NVCC_FLAGS" \
-DCUTLASS_NVCC_ARCHS:STRING="100a" \
-DCUTLASS_NVCC_KEEP:STRING="OFF" \
-DCUTLASS_ENABLE_F16C:STRING="ON" \
-DCUTLASS_LIBRARY_KERNELS:STRING="$CUTLASS_LIBRARY_KERNELS" \
-DCUTLASS_LIBRARY_IGNORE_KERNELS:STRING="$CUTLASS_LIBRARY_IGNORE_KERNELS" \
${COMPILER_CACHE_FLAGS} \
--no-warn-unused-cli \
-S. \
-B../build/
