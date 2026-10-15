#include "common.h"
#include "dequantize.h"

// TurboQuant4 bulk dequant to fp16 (for prefill FA)
kernel void kernel_turbo4_dequant_f16(
        device const block_turbo4_0 * src [[buffer(0)]],
        device half                 * dst [[buffer(1)]],
        constant int64_t            & n_blocks [[buffer(2)]],
        uint tgpig [[threadgroup_position_in_grid]],
        uint tiitg [[thread_index_in_threadgroup]],
        uint tptg  [[threads_per_threadgroup]]) {
    const int64_t blk_idx = tgpig * tptg + tiitg;
    if (blk_idx >= n_blocks) return;

    device const block_turbo4_0 * xb = src + blk_idx;
    device half * dst_blk = dst + blk_idx * 128;
    const float norm = float(xb->norm);

    for (int j = 0; j < 128; j++) {
        uint8_t idx = (xb->qs[j / 2] >> ((j % 2) * 4)) & 0xF;
        dst_blk[j] = half(turbo_centroids_4bit[idx] * norm);
    }
}

// In-place Fast Walsh-Hadamard Transform kernel for Metal
// Operates on contiguous groups of 128 float elements
kernel void kernel_turbo_wht(
        constant ggml_metal_kargs_turbo_wht & args [[buffer(0)]],
        device float * src [[buffer(1)]],
        device float * dst [[buffer(2)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint  tiitg [[thread_index_in_threadgroup]],
        uint3 tptg  [[threads_per_threadgroup]]) {
    const int64_t group_idx = tgpig.x * tptg.x + tiitg;
    const int64_t n_groups = args.n_elements / 128;

    if (group_idx >= n_groups) {
        return;
    }

    device const float * in_ptr = src + group_idx * 128;
    device float * out_ptr = dst + group_idx * 128;

    float x[128];
    for (int i = 0; i < 128; i++) {
        x[i] = in_ptr[i];
    }

    if (args.direction == 0) {
        turbo_rotate_forward(x, turbo_wht_signs1, turbo_wht_signs2);
    } else {
        turbo_rotate_inverse(x, turbo_wht_signs1, turbo_wht_signs2);
    }

    for (int i = 0; i < 128; i++) {
        out_ptr[i] = x[i];
    }
}
