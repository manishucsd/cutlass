### Define CUTLASS_LIBRARY_KERNELS strings for different GEMM types
BF16_BF16_TNN="\
s64x16x16gemm_bf16_bf16_f32_bf16_bf16*tnn*align8,\
s64x32x16gemm_bf16_bf16_f32_bf16_bf16*tnn*align8,\
s64x64x16gemm_bf16_bf16_f32_bf16_bf16*tnn*align8,\
s64x128x16gemm_bf16_bf16_f32_bf16_bf16*tnn*align8"

FE4M3_BF16_TNN="\
s64x32x16gemm_e4m3_bf16_f32_bf16_bf16*tnn*align16,\
s64x64x16gemm_e4m3_bf16_f32_bf16_bf16*tnn*align16,\
s64x128x16gemm_e4m3_bf16_f32_bf16_bf16*tnn*align16"

FE4M3_BF16_TNT="\
s64x32x16gemm_e4m3_bf16_f32_bf16_bf16*tnt*align16,\
s64x64x16gemm_e4m3_bf16_f32_bf16_bf16*tnt*align16,\
s64x128x16gemm_e4m3_bf16_f32_bf16_bf16*tnt*align16"

#### Parse the command-line arguments
GEMM_TYPE="bf16_bf16_tnn"

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
    bf16_bf16_tnn)
        CUTLASS_LIBRARY_KERNELS=$BF16_BF16_TNN
        ;;
    fe4m3_bf16_tnn)
        CUTLASS_LIBRARY_KERNELS=$FE4M3_BF16_TNN
        ;;
    fe4m3_bf16_tnt)
        CUTLASS_LIBRARY_KERNELS=$FE4M3_BF16_TNT
        ;;
    all)
        CUTLASS_LIBRARY_KERNELS="$BF16_BF16_TNN,$FE4M3_BF16_TNN,$FE4M3_BF16_TNT"
        ;;
    *)
        echo "Unknown GEMM_TYPE: $GEMM_TYPE"
        exit 1
        ;;
esac

### Set the CUTLASS_LIBRARY_IGNORE_KERNELS to ignore non-working issue-prone kernels
# Pingpong and stream_k kernels are ignored because of the following CUTLASS Issues:
# 1. https://github.com/NVIDIA/cutlass/issues/2152
# 2. https://github.com/NVIDIA/cutlass/issues/2121
CUTLASS_LIBRARY_IGNORE_KERNELS="\
gemm_grouped*,\
gemm_planar*,\
shfl,\
scl,\
sclzr,\
stream_k,\
"

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
-DCUTLASS_NVCC_ARCHS:STRING="90a" \
-DCUTLASS_NVCC_KEEP:STRING="OFF" \
-DCUTLASS_ENABLE_F16C:STRING="ON" \
-DCUTLASS_LIBRARY_INSTANTIATION_LEVEL:STRING="max" \
-DCUTLASS_LIBRARY_KERNELS:STRING="$CUTLASS_LIBRARY_KERNELS" \
-DCUTLASS_LIBRARY_IGNORE_KERNELS:STRING="$CUTLASS_LIBRARY_IGNORE_KERNELS" \
${COMPILER_CACHE_FLAGS} \
--no-warn-unused-cli \
-S. \
-B../build/
