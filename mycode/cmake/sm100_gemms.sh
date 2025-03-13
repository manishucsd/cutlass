### Define CUTLASS_LIBRARY_KERNELS strings for different GEMM types
BF16="\
cutlass3x_sm100_tensorop_s256x256x16gemm_bf16_bf16_f32_void_f32_256x256x64_2x1x1_0_tnt_align8_2sm"

FP8="\
cutlass3x_sm100_tensorop_s128x128x32gemm_f8_f8_f32_bf16_bf16_128x128x128_2x1x1_0_tnt_align16_2sm_epi_tma"

FE2M1="\
cutlass3x_sm100_tensorop_s256x256x32gemm_e2m1_e2m1_f32_void_f32_256x256x128_2x1x1_0_tnt_align128_2sm"

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
    all)
        CUTLASS_LIBRARY_KERNELS="$BF16,$FP8,$FE2M1"
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

### Run the cmake command
set -x

cmake \
-DCMAKE_BUILD_TYPE:STRING=Release \
-DCUTLASS_NVCC_ARCHS:STRING="100a" \
-DCUTLASS_NVCC_KEEP:STRING="OFF" \
-DCUTLASS_ENABLE_F16C:STRING="ON" \
-DCUTLASS_LIBRARY_KERNELS:STRING="$CUTLASS_LIBRARY_KERNELS" \
-DCUTLASS_LIBRARY_IGNORE_KERNELS:STRING="$CUTLASS_LIBRARY_IGNORE_KERNELS" \
${COMPILER_CACHE_FLAGS} \
--no-warn-unused-cli \
-S. \
-B../build/
