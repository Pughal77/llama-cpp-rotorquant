#pragma once

// TurboQuant Fast Walsh-Hadamard rotation for Metal
// Replaces 256KB dense matrices with 512 bytes of sign arrays + O(d log d) butterfly
// Generated with seed=42 (rotation) and seed=1042 (QJL)

// --- Rotation sign arrays ---
constant float turbo_wht_signs1[128] = {
    -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f};
constant float turbo_wht_signs2[128] = {
    1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f};

// --- QJL sign arrays ---
constant float turbo_qjl_wht_signs1[128] = {
    1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f};
constant float turbo_qjl_wht_signs2[128] = {
    1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f};

// --- Fast Walsh-Hadamard Transform (in-place, normalized) ---
// O(n log n) = 896 operations for n=128, vs O(n2) = 16384 for dense matvec
static void turbo_fwht_128(thread float * x) {
    for (int h = 1; h < 128; h *= 2) {
        for (int i = 0; i < 128; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j];
                float b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }
    // Normalize by 1/sqrt(128)
    const float inv_sqrt_128 = 0.08838834764831845f; // 1/sqrt(128)
    for (int i = 0; i < 128; i++) {
        x[i] *= inv_sqrt_128;
    }
}

// --- Forward rotation: signs1 -> FWHT -> signs2 ---
static void turbo_rotate_forward(thread float * x, constant float * s1, constant float * s2) {
    for (int i = 0; i < 128; i++) x[i] *= s1[i];
    turbo_fwht_128(x);
    for (int i = 0; i < 128; i++) x[i] *= s2[i];
}

// --- Inverse rotation: signs2 -> FWHT -> signs1 (FWHT is its own inverse) ---
static void turbo_rotate_inverse(thread float * x, constant float * s1, constant float * s2) {
    for (int i = 0; i < 128; i++) x[i] *= s2[i];
    turbo_fwht_128(x);
    for (int i = 0; i < 128; i++) x[i] *= s1[i];
}

// Pre-packed half4 sign arrays for vectorized WHT
constant half4 turbo_wht_signs1_h4[32] = {
    half4(-1.0h, 1.0h, 1.0h, -1.0h), half4(-1.0h, 1.0h, -1.0h, 1.0h),
    half4(-1.0h, -1.0h, 1.0h, 1.0h), half4(1.0h, 1.0h, 1.0h, 1.0h),
    half4(1.0h, -1.0h, 1.0h, -1.0h), half4(1.0h, -1.0h, -1.0h, 1.0h),
    half4(1.0h, 1.0h, -1.0h, 1.0h), half4(1.0h, -1.0h, -1.0h, -1.0h),
    half4(-1.0h, 1.0h, 1.0h, -1.0h), half4(1.0h, 1.0h, -1.0h, 1.0h),
    half4(-1.0h, 1.0h, 1.0h, -1.0h), half4(-1.0h, 1.0h, -1.0h, 1.0h),
    half4(1.0h, 1.0h, 1.0h, -1.0h), half4(-1.0h, -1.0h, -1.0h, -1.0h),
    half4(1.0h, -1.0h, 1.0h, 1.0h), half4(1.0h, 1.0h, -1.0h, 1.0h),
    half4(-1.0h, -1.0h, 1.0h, -1.0h), half4(-1.0h, -1.0h, 1.0h, -1.0h),
    half4(-1.0h, -1.0h, 1.0h, -1.0h), half4(-1.0h, -1.0h, 1.0h, 1.0h),
    half4(1.0h, -1.0h, -1.0h, 1.0h), half4(1.0h, 1.0h, -1.0h, -1.0h),
    half4(1.0h, 1.0h, -1.0h, 1.0h), half4(1.0h, -1.0h, 1.0h, -1.0h),
    half4(-1.0h, 1.0h, 1.0h, -1.0h), half4(1.0h, -1.0h, 1.0h, -1.0h),
    half4(1.0h, 1.0h, 1.0h, 1.0h), half4(-1.0h, 1.0h, -1.0h, 1.0h),
    half4(1.0h, -1.0h, 1.0h, 1.0h), half4(-1.0h, -1.0h, -1.0h, -1.0h),
    half4(-1.0h, 1.0h, 1.0h, -1.0h), half4(1.0h, 1.0h, -1.0h, 1.0h)
};
constant half4 turbo_wht_signs2_h4[32] = {
    half4(1.0h, 1.0h, 1.0h, 1.0h), half4(-1.0h, 1.0h, 1.0h, -1.0h),
    half4(1.0h, -1.0h, -1.0h, -1.0h), half4(1.0h, -1.0h, -1.0h, -1.0h),
    half4(1.0h, 1.0h, -1.0h, -1.0h), half4(1.0h, -1.0h, 1.0h, -1.0h),
    half4(1.0h, -1.0h, -1.0h, 1.0h), half4(-1.0h, 1.0h, 1.0h, 1.0h),
    half4(1.0h, 1.0h, -1.0h, -1.0h), half4(-1.0h, 1.0h, -1.0h, -1.0h),
    half4(-1.0h, -1.0h, -1.0h, -1.0h), half4(1.0h, 1.0h, 1.0h, -1.0h),
    half4(1.0h, -1.0h, 1.0h, 1.0h), half4(1.0h, -1.0h, -1.0h, 1.0h),
    half4(-1.0h, -1.0h, -1.0h, -1.0h), half4(-1.0h, -1.0h, 1.0h, 1.0h),
    half4(1.0h, -1.0h, 1.0h, -1.0h), half4(-1.0h, -1.0h, -1.0h, 1.0h),
    half4(-1.0h, 1.0h, -1.0h, 1.0h), half4(-1.0h, -1.0h, 1.0h, 1.0h),
    half4(-1.0h, 1.0h, -1.0h, 1.0h), half4(1.0h, -1.0h, 1.0h, -1.0h),
    half4(-1.0h, -1.0h, -1.0h, 1.0h), half4(-1.0h, -1.0h, 1.0h, -1.0h),
    half4(1.0h, -1.0h, 1.0h, 1.0h), half4(1.0h, -1.0h, -1.0h, 1.0h),
    half4(-1.0h, 1.0h, -1.0h, 1.0h), half4(1.0h, -1.0h, -1.0h, 1.0h),
    half4(-1.0h, 1.0h, -1.0h, 1.0h), half4(1.0h, -1.0h, 1.0h, -1.0h),
    half4(1.0h, -1.0h, -1.0h, -1.0h), half4(-1.0h, -1.0h, 1.0h, -1.0h)
};

static void turbo_fwht_128_half4(thread half4 * v) {
    for (int i = 0; i < 32; i++) {
        half4 a = v[i];
        v[i] = half4(a.x + a.y, a.x - a.y, a.z + a.w, a.z - a.w);
    }
    for (int i = 0; i < 32; i++) {
        half4 a = v[i];
        v[i] = half4(a.x + a.z, a.y + a.w, a.x - a.z, a.y - a.w);
    }
    for (int h = 4; h < 128; h *= 2) {
        int vec_stride = h / 4;
        for (int i = 0; i < 32; i++) {
            int group_pos = i % (2 * vec_stride);
            if (group_pos < vec_stride) {
                int partner = i + vec_stride;
                half4 a = v[i];
                half4 b = v[partner];
                v[i]       = a + b;
                v[partner] = a - b;
            }
        }
    }
    const half4 inv_sqrt_128 = half4(0.08838834764831845h);
    for (int i = 0; i < 32; i++) {
        v[i] *= inv_sqrt_128;
    }
}
