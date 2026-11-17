# HIP / ROCm Build Guide and Environmental Fixes

## System Environment

- **Operating System**: Linux (Arch Linux x86_64)
- **GPU**: AMD Radeon RX 6600 (Navi 23, Architecture: `gfx1032`, 8 GB VRAM)
- **ROCm Version**: 7.2.4
- **HIP Compiler**: AMD Clang 22.0.0 (`/opt/rocm/lib/llvm/bin/clang++`)
- **ROCm Device Bitcode**: `/opt/rocm/amdgcn/bitcode`

---

## Backend Selection: Vulkan vs. HIP

The user instruction requested testing Vulkan first, falling back to HIP / hipBLAS:

### Why Vulkan Failed on this Setup
Upstream commit `3bc612fe2` (PR #31416) introduced a mandatory requirement in `ggml/src/ggml-vulkan/CMakeLists.txt`:
```cmake
find_package(SPIRV-Headers CONFIG REQUIRED)
```
While `spirv-tools` was present on the machine, Arch Linux separates `spirv-tools` and `spirv-headers`. CMake could not locate `SPIRV-HeadersConfig.cmake`. Furthermore, Vulkan does not currently implement custom FlashAttention kernels for TurboQuant or RotorQuant.

### Selecting HIP / hipBLAS
ROCm 7.2.4 was fully installed at `/opt/rocm`, and the repository contains dedicated HIP/CUDA implementations for FlashAttention and quantization. HIP was configured and built targeting `gfx1032`.

---

## Environmental and Code Fixes

Building with HIP required several targeted fixes across headers, templates, hardware constraints, and build scripts.

### 1. Header Portability in CUDA / HIP Headers
- **Problem**: `ggml/src/ggml-cuda/cpy-planar-iso.cuh` included `<cuda_runtime.h>`, and `ggml/src/ggml-cuda/set-rows-planar-iso.cuh` included `<cuda_fp16.h>`. When building under ROCm with the HIP compiler, these NVIDIA-specific headers do not exist, causing fatal include errors.
- **Fix**: Replaced direct CUDA header inclusions with `"common.cuh"`. When `GGML_USE_HIP` is defined, `common.cuh` automatically delegates to `vendors/hip.h`, which pulls in `<hip/hip_runtime.h>` and `<hip/hip_fp16.h>`.

### 2. Elimination of Invalid Template Specializations in `set-rows.cu`
- **Problem**: The quantizer dispatch block for TurboQuant, PlanarQuant, and IsoQuant had been copy-pasted into the explicit specializations `set_rows_cuda<half, int32_t>` and `set_rows_cuda<half, int64_t>`. This produced two compiler errors:
  1. `error: unknown type name 'idx_t'`: `idx_t` is a template parameter only in the general template, not in explicit `half` specializations.
  2. `error: no matching function for call to 'set_rows_cuda_quant'`: Candidate function expected `const float *`, but received `const half *`.
- **Fix**: Removed the misplaced quantizer blocks from `set_rows_cuda<half, ...>`. As governed by `GGML_ASSERT(src0->type == GGML_TYPE_F32 || (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16))`, half-precision source tensors only target `GGML_TYPE_F16`.

### 3. Hardware Shared Memory (LDS) Limit on AMD RDNA2
- **Problem**: During compilation of `ggml/src/ggml-cuda/template-instances/fattn-tile-instance-dkq640-dv512.cu`, the compiler threw:
  ```
  error: local memory (67584) exceeds limit (65536) in 'void flash_attn_tile<640, 512, 2, 16, true>(...)'
  ```
  AMD RDNA2 GPUs (`gfx1032`) possess a strict hardware Local Data Share (LDS) ceiling of 64 KB (65,536 bytes) per workgroup. The large tile configuration with head dimensions $D_{KQ}=640, D_V=512$ required 67,584 bytes of shared memory.
- **Fix**:
  1. In `ggml/src/ggml-cuda/template-instances/fattn-tile-instance-dkq640-dv512.cu`, wrapped `DECL_FATTN_TILE_CASE(640, 512)` with `#if !defined(GGML_USE_HIP)`.
  2. In `ggml/src/ggml-cuda/fattn-tile.cu`, added a `#if !defined(GGML_USE_HIP)` guard around the runtime dispatch case `640`, replacing it with `GGML_ABORT` explaining the LDS hardware limit under HIP.

### 4. Linkage Specification in `ggml-cpu/ops.cpp`
- **Problem**: In `ggml/src/ggml-cpu/ops.cpp`:
  ```cpp
  extern "C" GGML_API int turbo3_cpu_wht_group_size;
  ```
  Because `GGML_API` already expands to `__attribute__ ((visibility ("default"))) extern`, this resulted in `extern "C" extern int ...`, which violates C++ linkage specifications.
- **Fix**: Replaced the declaration with block linkage syntax:
  ```cpp
  extern "C" { GGML_API int turbo3_cpu_wht_group_size; }
  ```

### 5. Inclusion of Turbo and Rotor Template Instances in HIP CMake
- **Problem**: `ggml/src/ggml-hip/CMakeLists.txt` previously only compiled the upstream quantization combinations for FlashAttention vectors (`f16-f16`, `q4_0-q4_0`, `q8_0-q8_0`, `bf16-bf16`, `f8_e4m3-f8_e4m3`). Running FlashAttention with `turbo` or `planar` KV cache under HIP would fail at runtime due to missing symbols.
- **Fix**: Appended all 42 custom template instances (`fattn-vec-instance-turbo*`, `fattn-vec-instance-planar*`, `fattn-vec-instance-iso*`) to `GGML_SOURCES_ROCM` in `ggml/src/ggml-hip/CMakeLists.txt`.

### 6. Dynamic Runtime Library Resolution
- **Problem**: Binaries failed on startup with:
  ```
  error while loading shared libraries: libhipblas.so.3: cannot open shared object file: No such file or directory
  ```
- **Fix**: Exported the ROCm library directory to `LD_LIBRARY_PATH`:
  ```bash
  export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
  ```

---

## Configuration and Build Commands

To reproduce the working HIP build:

```bash
# 1. Clean previous build artifacts
rm -rf build

# 2. Configure CMake with HIP targeting gfx1032
HIPCXX="/opt/rocm/lib/llvm/bin/clang++" \
HIP_PATH="/opt/rocm" \
HIP_DEVICE_LIB_PATH="/opt/rocm/amdgcn/bitcode" \
cmake -B build \
    -DGGML_HIP=ON \
    -DAMDGPU_TARGETS=gfx1032 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH=/opt/rocm

# 3. Build binaries using multiple threads
cmake --build build --config Release -j12
```

---

## Execution and Verification

### 1. Verify Available ROCm Devices
```bash
LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH ./build/bin/llama-cli --list-devices
```
Output:
```
Available devices:
  ROCm0: AMD Radeon RX 6600 (8176 MiB, 8094 MiB free)
  ROCm1: AMD Ryzen 9 7900 12-Core Processor (15612 MiB, 15588 MiB free)
```

### 2. Run Quantization Unit Tests
```bash
LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH ./build/bin/test-quantize-fns
```
Tests pass for all 43 quantization formats, including:
- `turbo2`, `turbo3`, `turbo4`
- `planar3`, `planar4`
- `iso3`, `iso4`

### 3. Load Model with Full GPU Offload
```bash
LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH ./build/bin/llama-cli \
    -m /home/raijinshounen/myRoot/mySEJourney/model-inference/models/Qwen3.5-4B-Q4_0.gguf \
    -ngl 99 \
    -p "Explain what a rotor is in geometric algebra in one sentence." \
    -n 32 \
    --simple-io
```
Offload verification from logs:
```
load_tensors: offloading output layer to GPU
load_tensors: offloading 31 repeating layers to GPU
load_tensors: offloaded 33/33 layers to GPU
load_tensors:   ROCm0 model buffer size =  2453.09 MiB
llama_kv_cache: ROCm0 KV buffer size    =  4240.00 MiB
llama_memory_recurrent: ROCm0 RS buffer size = 50.25 MiB
sched_reserve:  ROCm0 compute buffer size    = 206.52 MiB
```
All model weights, recurrent states, and KV cache buffers reside directly on `ROCm0`.
