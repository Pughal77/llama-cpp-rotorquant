# TurboQuant & RotorQuant Merge Documentation

## Overview

This document details the merge between upstream `llama.cpp` (`master`, commit `ef6bc6f36`) and the `rebase/upstream-turbo-rotor` branch, which incorporates:
- **TurboQuant**: Randomized Walsh-Hadamard Transform (WHT) KV cache compression (`turbo2_0`, `turbo3_0`, `turbo4_0`).
- **RotorQuant**: 2D Givens planar rotations (`planar3_0`, `planar4_0`) and 4D quaternion isoclinic rotations (`iso3_0`, `iso4_0`).
- **Deferred Key Quantization**: Retention of keys in F16 during prompt processing (prefill) with bulk in-place GPU quantization on the first decoding token.

The collaborator repository (`johndpope/llama-cpp-turboquant`, branch `feature/planarquant-kv-cache`) was checked and verified to have no newer commits beyond our base commit (`a8b815339`).

---

## Technical Foundations: TurboQuant vs. RotorQuant

### 1. TurboQuant
TurboQuant addresses quantization outliers by rotating the KV vectors with a randomized Walsh-Hadamard Transform before uniform quantization:

$$R_{\text{WHT}} = D_2 \frac{1}{\sqrt{d}} H_d D_1$$

Where:
- $H_d$ is the Sylvester-Hadamard matrix of order $d=128$.
- $D_1, D_2 \in \{-1, +1\}^d$ are diagonal sign matrices chosen at initialization.
- Matrix multiplication complexity is $O(d \log d)$ with $d \times d$ dense mixing.

Because the rotation mixes all $d$ components:
- Queries $Q$ must be pre-rotated prior to attention: $\langle R Q, R K \rangle = \langle Q, K \rangle$.
- Attention values $V$ are rotated, so the attention output satisfies:
  $$\sum_t \alpha_t (R V_t) = R \left( \sum_t \alpha_t V_t \right)$$
- An explicit inverse WHT rotation ($R_{\text{WHT}}^{-1} = R_{\text{WHT}}^T$) must be evaluated on the attention output tensor via `ggml_turbo_wht`.

### 2. PlanarQuant (RotorQuant 2D)
PlanarQuant replaces dense $d$-dimensional mixing with pairwise 2D Givens rotations:
For each coordinate pair $(x_{2k}, x_{2k+1})$ where $k \in \{0, \dots, d/2 - 1\}$:

$$\begin{bmatrix} y_{2k} \\ y_{2k+1} \end{bmatrix} = \begin{bmatrix} \cos \theta_k & -\sin \theta_k \\ \sin \theta_k & \cos \theta_k \end{bmatrix} \begin{bmatrix} x_{2k} \\ x_{2k+1} \end{bmatrix}$$

- Computational cost: Exactly 4 fused multiply-adds (FMAs) per 2D pair.
- Disjoint Block Structure: Because each coordinate pair transforms independently without cross-pair mixing, the inverse rotation matrix is simply the 2D transpose:
  $$\begin{bmatrix} \cos \theta_k & \sin \theta_k \\ -\sin \theta_k & \cos \theta_k \end{bmatrix}$$
- On-the-Fly Inversion: Dequantization inside FlashAttention kernels (`vec_dot_KQ` and `dequantize_V`) applies the inverse rotation directly during memory loads. This eliminates the need to pre-rotate $Q$ or post-rotate attention output in the compute graph.

### 3. IsoQuant (RotorQuant 4D)
IsoQuant extends 2D rotations to 4D coordinate blocks using isoclinic (left quaternion) multiplication:
For each 4-tuple $(x_{4k}, x_{4k+1}, x_{4k+2}, x_{4k+3})$:

$$y_g = q_g \cdot x_g$$

Where $q_g$ is a unit quaternion ($q_g \in \mathbb{H}, \|q_g\| = 1$).
- Computational cost: 16 FMAs per 4D group.
- Inversion: Inverted inside FlashAttention kernels via conjugate quaternion multiplication: $x_g = q_g^* \cdot y_g$.

### 4. Deferred Quantization
Prefill tokens (`ubatch.n_tokens > 1`) retain keys in uncompressed `GGML_TYPE_F16` to prevent error accumulation across long prompt contexts. When generation switches to autoregressive decode (`ubatch.n_tokens == 1`), `convert_deferred_keys()` runs parallel GPU kernels (`cpy-planar-iso.cu` or `set-rows.cu`) to rotate, quantize, and pack the prompt keys in place.

---

## Merge Conflicts and Resolution Details

The rebase/merge against upstream `ef6bc6f36` created conflicts across 19 files:

### 1. Type and Op Definitions (`ggml/include/ggml.h`, `ggml/include/ggml-rpc.h`, `ggml/src/ggml.c`)
- **GGML Type Table**: Preserved custom type enum entries:
  - `GGML_TYPE_TURBO3_0` (44)
  - `GGML_TYPE_TURBO4_0` (45)
  - `GGML_TYPE_TURBO2_0` (46)
  - `GGML_TYPE_PLANAR3_0` (47)
  - `GGML_TYPE_ISO3_0` (48)
  - `GGML_TYPE_PLANAR4_0` (49)
  - `GGML_TYPE_ISO4_0` (50)
  - Updated `GGML_TYPE_COUNT = 51`.
- **Operator Table**: Added upstream `GGML_OP_TOP_K_MASK` while preserving `GGML_OP_TURBO_WHT`.
- **Operator Count**: Updated `GGML_OP_COUNT = 100` and aligned `static_assert(GGML_OP_COUNT == 100)` across `ggml.c` and `ggml-rpc.h`.
- **Type Traits**: Preserved `type_traits` entries for all 7 turbo and rotor quantization types (type names, block sizes, quantize/dequantize function pointers).

### 2. CPU Backend (`ggml/src/ggml-cpu/ggml-cpu.c`, `ops.h`, `ops.cpp`)
- Added forward operations for both `ggml_compute_forward_top_k_mask` and `ggml_compute_forward_turbo_wht`.
- Fixed C++ linkage error in `ggml-cpu/ops.cpp` for the global variable `turbo3_cpu_wht_group_size` by wrapping it in `extern "C" { GGML_API int turbo3_cpu_wht_group_size; }`.

### 3. CUDA & HIP FlashAttention (`ggml/src/ggml-cuda/`)
- `convert.cu`: Added FP8 alongside TurboQuant and RotorQuant type conversion cases.
- `fattn-common.cuh`: Integrated FP8 and Turbo/Planar/Iso types into `get_vec_dot_KQ()` and `get_dequantize_V()`.
- `fattn-mma-f16.cuh`: Preserved tile head configurations (256/256, 640/512).
- `fattn-vec.cuh`: Preserved `K_is_unquantized` checks for deferred key states, `Q_q8_1` quantization, and vector kernel declarations.
- `fattn.cu`: Added turbo and planar type branches to KV type validation checks.
- `vendors/hip.h`: Updated shuffle intrinsic macros to handle both 3-argument and 4-argument call patterns under ROCm.

### 4. Apple Metal Backend (`ggml/src/ggml-metal/`)
- `CMakeLists.txt`: Retained compilation of `kernels/fa_turbo.metal` and `kernels/turbo_wht.metal`.
- `ggml-metal-device.m`: Preserved pipeline states for `turbo_wht` and FlashAttention turbo kernels.
- `ggml-metal-impl.h` & `ggml-metal-ops.cpp`: Integrated turbo graph nodes and dispatch rules.

### 5. High-Level Context and Attention Graph (`src/`)
- `llama-context.cpp`: Merged upstream FP8 KV cache validation while ensuring TurboQuant and RotorQuant KV cache types auto-enable FlashAttention (`cparams.flash_attn = true`).
- `llama-graph.cpp`: Preserved all three `build_attn` hooks:
  - Pre-rotating $Q$ with `ggml_turbo_wht` when `cparams.turbo_rotation` is active.
  - Passing `kv_rows` tensor to `build_attn_mha`.
  - Applying inverse WHT post-rotation on attention output tensor.
- `llama-kv-cache.h` & `llama-kv-cache.cpp`: Merged per-layer scale storage (`k_scale`, `v_scale`) and guarded `turbo_rotation` allocation to avoid allocations when `ctxs_bufs.no_alloc` is true.

---

## Verification

1. **Compilation**: Built with CMake using HIP on AMD Radeon RX 6600 (`gfx1032`) without compilation errors.
2. **Quantization Tests**: Ran `./build/bin/test-quantize-fns` covering all 43 quantization formats; all passed including `turbo2/3/4`, `planar3/4`, and `iso3/4`.
3. **End-to-End Inference**: Ran `./build/bin/llama-cli` on `Qwen3.5-4B-Q4_0.gguf` with all 33/33 layers fully offloaded to GPU VRAM (`ROCm0`), verifying generation correctness.
