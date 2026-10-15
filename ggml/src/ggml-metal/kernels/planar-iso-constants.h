#pragma once

#include "turbo-quant.h"

constant float planar_centroids_3bit[8] = {
    -0.1906850000f, -0.1178320000f, -0.0657170000f, -0.0214600000f, 0.0214600000f, 0.0657170000f, 0.1178320000f, 0.1906850000f,
};

// Rotation parameters (seed=42, 64 cos/sin pairs for d=128)
constant float planar_cos_64[64] = {
    0.7386546135f, 0.8607548475f, -0.7411674857f, 0.9674890637f, -0.7723053098f, -0.8056974411f, -0.0412844308f, 0.2707833052f,
    0.9315500855f, 0.6698185802f, 0.9167487621f, -0.8320636749f, 0.6818146110f, -0.9108457565f, -0.0559285842f, -0.9032276273f,
    0.7519487143f, -0.8941103816f, -0.1039871648f, -0.6961420774f, -0.1230370328f, -0.9328963161f, -0.2905603051f, 0.4910068214f,
    0.7889407277f, -0.1221836656f, -0.6316579580f, 0.3128163815f, -0.9563610554f, 0.9992509484f, 0.9540294409f, 0.8902468085f,
    0.7543080449f, -0.8664138913f, -0.5232898593f, 0.3621287644f, -0.8825117350f, 0.8234673142f, -0.9416025877f, -0.5480425358f,
    -0.6644080281f, -0.6585279703f, -0.2460795939f, 0.9438471198f, 0.2427810431f, -0.1960992366f, 0.2403578013f, -0.8461306095f,
    0.0246123374f, 0.3372744620f, 0.9994974732f, -0.3494733870f, 0.7438930869f, 0.8452339768f, -0.6177822948f, -0.2662552595f,
    -0.5457068086f, -0.9985070229f, 0.7757105827f, 0.6141811609f, -0.9805000424f, 0.5425475240f, -0.5663578510f, -0.4696439803f,
};
constant float planar_sin_64[64] = {
    -0.6740840673f, -0.5090196729f, 0.6713201404f, -0.2529129684f, 0.6352515221f, -0.5923272967f, 0.9991474152f, -0.9626403451f,
    -0.3636130989f, 0.7425247431f, -0.3994642496f, -0.5546801090f, -0.7315250039f, -0.4127469361f, -0.9984347820f, 0.4291617870f,
    -0.6592215896f, -0.4478466809f, 0.9945786595f, -0.7179040313f, 0.9924020767f, 0.3601450622f, 0.9568566680f, -0.8711557388f,
    0.6144692898f, 0.9925075173f, 0.7752471566f, 0.9498136044f, -0.2921875417f, 0.0386975110f, -0.2997128963f, 0.4554784000f,
    -0.6565206647f, -0.4993265271f, 0.8521547318f, -0.9321280718f, -0.4702904224f, -0.5673637390f, -0.3367263079f, 0.8364504576f,
    -0.7473700047f, 0.7525562644f, -0.9692496061f, -0.3303825557f, -0.9700810909f, 0.9805840850f, -0.9706843495f, -0.5329755545f,
    -0.9996970892f, 0.9414063692f, 0.0316982083f, 0.9369462729f, 0.6682986617f, -0.5343964100f, -0.7863491774f, -0.9639025331f,
    -0.8379761577f, 0.0546237342f, -0.6310887933f, 0.7891650796f, -0.1965190321f, 0.8400250673f, -0.8241594434f, 0.8828558922f,
};


// IsoQuant quaternion rotation parameters (seed=42, 32 unit quaternions for d=128)
constant float iso_qw_32[32] = {
    0.5765609741f, 0.3176580369f, -0.3234235942f, -0.5127438903f, 0.9233905673f, -0.3323571086f, 0.5468608141f, -0.2500519454f,
    -0.5812215805f, 0.3228830695f, -0.7299832702f, -0.4535493255f, -0.7338157296f, -0.2884652913f, -0.9000198841f, -0.0377033800f,
    0.5104404092f, 0.2033989877f, -0.2462528497f, 0.2314069420f, 0.0072374810f, 0.3923372924f, 0.4958070219f, -0.7235037088f,
    -0.9383618832f, 0.4430379272f, -0.2075705230f, 0.1983736306f, -0.8834578991f, 0.7389573455f, -0.0156172011f, 0.7738668919f,
};
constant float iso_qx_32[32] = {
    0.4450169504f, -0.5780548453f, 0.7089627385f, -0.3940812945f, -0.0897334740f, 0.4727236331f, 0.5542563796f, 0.0450818054f,
    -0.3657043576f, -0.4298477769f, 0.4666220546f, 0.7556306720f, -0.5284956098f, 0.7042509317f, 0.0230921544f, 0.7110687494f,
    0.3024962246f, -0.1157865301f, 0.7490812540f, -0.2582575679f, -0.2255804837f, 0.3838746250f, -0.3209520578f, -0.3477301002f,
    0.1824720055f, 0.4032751918f, 0.8433781862f, 0.9533935785f, -0.0620501526f, 0.0927560627f, 0.2964956462f, 0.2402082384f,
};
constant float iso_qy_32[32] = {
    0.2695076466f, -0.0201656222f, -0.1687686443f, -0.5415957570f, -0.2796611190f, 0.3510629535f, 0.2609911859f, -0.2715902030f,
    -0.0937586129f, 0.3095585108f, -0.4123268127f, -0.4394895136f, 0.0626545250f, -0.4811822474f, -0.0407132693f, -0.4566248953f,
    0.7834537029f, -0.6187923551f, 0.0809760988f, -0.8879503012f, -0.8928058147f, 0.8350352049f, -0.6994170547f, 0.5606835485f,
    0.2933705449f, 0.7377059460f, 0.4534837306f, -0.0009816211f, -0.3632916510f, -0.3959124386f, 0.1631654203f, 0.5088164806f,
};
constant float iso_qz_32[32] = {
    -0.6300023794f, -0.7513582706f, -0.6035611629f, 0.5370919704f, 0.2471584976f, 0.7367672324f, 0.5706370473f, 0.9282674193f,
    0.7208684087f, -0.7843156457f, -0.2817355990f, -0.1736787707f, 0.4222335219f, -0.4350655377f, 0.4333281815f, 0.5333415866f,
    0.1847889870f, 0.7498788238f, 0.6096553802f, -0.3021556735f, -0.3898189068f, 0.0377884321f, 0.4024685621f, 0.2031257302f,
    0.0107116764f, -0.3112498820f, 0.1999502629f, -0.2273492515f, 0.2892593443f, 0.5372074246f, 0.9408631325f, 0.2907505929f,
};

// ===== iso4 and planar4: aliases for turbo4 dequantize (same block layout) =====
// The CPU quantize path uses turbo4's WHT rotation for both.
// Custom Metal set_rows kernels with quaternion/Givens rotation are future work.

// planar4 dequantize: nibble unpack -> centroid lookup -> inverse Givens rotation
template <typename type4x4>
void dequantize_planar4_0(device const block_planar4_0 * xb, short il, thread type4x4 & reg) {
    const float norm = float(xb->norm);
    const int base_elem = il * 16;
    float4x4 reg_f;

    for (int g = 0; g < 4; g++) {
        float4 raw;
        for (int k = 0; k < 4; k++) {
            const int j = base_elem + g * 4 + k;
            uint8_t idx = (xb->qs[j / 2] >> ((j % 2) * 4)) & 0xF;
            raw[k] = turbo_centroids_4bit[idx];
        }

        // Inverse Givens rotation on pairs
        int pair0 = (base_elem + g * 4) / 2;
        int pair1 = pair0 + 1;

        float c0 = planar_cos_64[pair0], s0 = planar_sin_64[pair0];
        float c1 = planar_cos_64[pair1], s1 = planar_sin_64[pair1];

        float f0 =  c0 * raw[0] + s0 * raw[1];
        float f1 = -s0 * raw[0] + c0 * raw[1];
        float f2 =  c1 * raw[2] + s1 * raw[3];
        float f3 = -s1 * raw[2] + c1 * raw[3];

        reg_f[g] = float4(f0 * norm, f1 * norm, f2 * norm, f3 * norm);
    }
    reg = (type4x4) reg_f;
}

template <typename type4>
void dequantize_planar4_0_t4(device const block_planar4_0 * xb, short il, thread type4 & reg) {
    const float norm = float(xb->norm);
    const device uint8_t * qs = xb->qs + il * 2;
    const uint8_t qb0 = qs[0];
    const uint8_t qb1 = qs[1];

    float4 raw = float4(
        turbo_centroids_4bit[(qb0     ) & 0xF],
        turbo_centroids_4bit[(qb0 >> 4) & 0xF],
        turbo_centroids_4bit[(qb1     ) & 0xF],
        turbo_centroids_4bit[(qb1 >> 4) & 0xF]
    );

    // Inverse Givens rotation
    int pair0 = il * 2;  // each il covers 4 elements = 2 pairs
    int pair1 = pair0 + 1;

    float c0 = planar_cos_64[pair0], s0 = planar_sin_64[pair0];
    float c1 = planar_cos_64[pair1], s1 = planar_sin_64[pair1];

    float f0 =  c0 * raw[0] + s0 * raw[1];
    float f1 = -s0 * raw[0] + c0 * raw[1];
    float f2 =  c1 * raw[2] + s1 * raw[3];
    float f3 = -s1 * raw[2] + c1 * raw[3];

    reg = (type4) float4(f0 * norm, f1 * norm, f2 * norm, f3 * norm);
}

// iso4 dequantize: nibble unpack -> centroid lookup -> inverse quaternion rotation
template <typename type4x4>
void dequantize_iso4_0(device const block_iso4_0 * xb, short il, thread type4x4 & reg) {
    const float norm = float(xb->norm);
    const int base_elem = il * 16;
    float4x4 reg_f;

    for (int g = 0; g < 4; g++) {
        float4 raw;
        for (int k = 0; k < 4; k++) {
            const int j = base_elem + g * 4 + k;
            uint8_t idx = (xb->qs[j / 2] >> ((j % 2) * 4)) & 0xF;
            raw[k] = turbo_centroids_4bit[idx];
        }

        // Inverse quaternion rotation: conj(q_L) * v
        int qg = (base_elem + g * 4) / 4;  // quaternion group index
        float qw = iso_qw_32[qg], qx = -iso_qx_32[qg], qy = -iso_qy_32[qg], qz = -iso_qz_32[qg];
        float vw = raw[0], vx = raw[1], vy = raw[2], vz = raw[3];

        float rw = qw*vw - qx*vx - qy*vy - qz*vz;
        float rx = qw*vx + qx*vw + qy*vz - qz*vy;
        float ry = qw*vy - qx*vz + qy*vw + qz*vx;
        float rz = qw*vz + qx*vy - qy*vx + qz*vw;

        reg_f[g] = float4(rw * norm, rx * norm, ry * norm, rz * norm);
    }
    reg = (type4x4) reg_f;
}

template <typename type4>
void dequantize_iso4_0_t4(device const block_iso4_0 * xb, short il, thread type4 & reg) {
    const float norm = float(xb->norm);
    const device uint8_t * qs = xb->qs + il * 2;
    const uint8_t qb0 = qs[0];
    const uint8_t qb1 = qs[1];

    float4 raw = float4(
        turbo_centroids_4bit[(qb0     ) & 0xF],
        turbo_centroids_4bit[(qb0 >> 4) & 0xF],
        turbo_centroids_4bit[(qb1     ) & 0xF],
        turbo_centroids_4bit[(qb1 >> 4) & 0xF]
    );

    // Inverse quaternion rotation
    int qg = il;  // each il = 4 elements = 1 quaternion group
    float qw = iso_qw_32[qg], qx = -iso_qx_32[qg], qy = -iso_qy_32[qg], qz = -iso_qz_32[qg];

    float rw = qw*raw[0] - qx*raw[1] - qy*raw[2] - qz*raw[3];
    float rx = qw*raw[1] + qx*raw[0] + qy*raw[3] - qz*raw[2];
    float ry = qw*raw[2] - qx*raw[3] + qy*raw[0] + qz*raw[1];
    float rz = qw*raw[3] + qx*raw[2] - qy*raw[1] + qz*raw[0];

    reg = (type4) float4(rw * norm, rx * norm, ry * norm, rz * norm);
}

// planar4/iso4 set_rows: delegate to turbo4 (WHT rotation) for now
// Custom rotation kernels would be added here for differentiated quality

// ===== IsoQuant 3-bit dequantize (quaternion 4D block rotation) =====
// 4D blocks rotated by unit quaternion: T(v) = q_L * v
// Inverse: T^-1(v) = conj(q_L) * v
// 16 FMAs per quaternion multiply, 32 groups for d=128

constant float iso_centroids_3bit[8] = {
    -0.1906850000f, -0.1178320000f, -0.0657170000f, -0.0214600000f, 0.0214600000f, 0.0657170000f, 0.1178320000f, 0.1906850000f,
};



// Non-vec dequantize: 16 elements per call (il in {0..NL_ISO3-1})
template <typename type4x4>
void dequantize_iso3_0(device const block_iso3_0 * xb, short il, thread type4x4 & reg) {
    const float norm = float(xb->norm);
    const int qs_off = il * 4;
    float4x4 reg_f;

    for (int g = 0; g < 4; g++) {
        const uint8_t qb = xb->qs[qs_off + g];
        const uint8_t sb = xb->signs[il * 2 + g / 2];
        const int sshift = (g & 1) * 4;

        // Unpack 4 3-bit indices to centroid values
        float4 raw;
        raw[0] = iso_centroids_3bit[(qb & 0x03)        | (((sb >> (sshift + 0)) & 1) << 2)];
        raw[1] = iso_centroids_3bit[((qb >> 2) & 0x03) | (((sb >> (sshift + 1)) & 1) << 2)];
        raw[2] = iso_centroids_3bit[((qb >> 4) & 0x03) | (((sb >> (sshift + 2)) & 1) << 2)];
        raw[3] = iso_centroids_3bit[((qb >> 6) & 0x03) | (((sb >> (sshift + 3)) & 1) << 2)];

        // Inverse quaternion rotation on 4D group
        // Element indices within block: il*16 + g*4 + {0,1,2,3}
        // Quaternion group index: (il*16 + g*4) / 4 = il*4 + g
        int qg = il * 4 + g;

        // conj(q) * v: Hamilton product with conjugate quaternion
        float qw = iso_qw_32[qg], qx = -iso_qx_32[qg], qy = -iso_qy_32[qg], qz = -iso_qz_32[qg];
        float vw = raw[0], vx = raw[1], vy = raw[2], vz = raw[3];

        float rw = qw*vw - qx*vx - qy*vy - qz*vz;
        float rx = qw*vx + qx*vw + qy*vz - qz*vy;
        float ry = qw*vy - qx*vz + qy*vw + qz*vx;
        float rz = qw*vz + qx*vy - qy*vx + qz*vw;

        reg_f[g] = float4(rw * norm, rx * norm, ry * norm, rz * norm);
    }
    reg = (type4x4) reg_f;
}

// Vec dequantize: 4 elements per call (il in {0..NL_ISO3_VEC-1})
template <typename type4>
void dequantize_iso3_0_t4(device const block_iso3_0 * xb, short il, thread type4 & reg) {
    const float norm = float(xb->norm);
    const uint8_t qb = xb->qs[il];
    const uint8_t sb = xb->signs[il / 2];
    const int sshift = (il & 1) * 4;

    float4 raw;
    raw[0] = iso_centroids_3bit[(qb & 0x03)        | (((sb >> (sshift + 0)) & 1) << 2)];
    raw[1] = iso_centroids_3bit[((qb >> 2) & 0x03) | (((sb >> (sshift + 1)) & 1) << 2)];
    raw[2] = iso_centroids_3bit[((qb >> 4) & 0x03) | (((sb >> (sshift + 2)) & 1) << 2)];
    raw[3] = iso_centroids_3bit[((qb >> 6) & 0x03) | (((sb >> (sshift + 3)) & 1) << 2)];

    // Quaternion group = il (since each group is 4 elements)
    int qg = il;
    float qw = iso_qw_32[qg], qx = -iso_qx_32[qg], qy = -iso_qy_32[qg], qz = -iso_qz_32[qg];
    float vw = raw[0], vx = raw[1], vy = raw[2], vz = raw[3];

    float rw = qw*vw - qx*vx - qy*vy - qz*vz;
    float rx = qw*vx + qx*vw + qy*vz - qz*vy;
    float ry = qw*vy - qx*vz + qy*vw + qz*vx;
    float rz = qw*vz + qx*vy - qy*vx + qz*vw;

    reg = (type4) float4(rw * norm, rx * norm, ry * norm, rz * norm);
}


// IsoQuant set_rows: quaternion 4D rotation
// ===== PlanarQuant 3-bit dequantize (2D Givens rotation) =====
// Same block layout as turbo3, but inverse rotation is per-pair Givens.
// Much simpler than WHT: only 4 FMAs per pair, no full-block dependency.


// Non-vec dequantize: 16 elements per call (il in {0..NL_PLANAR3-1})
template <typename type4x4>
void dequantize_planar3_0(device const block_planar3_0 * xb, short il, thread type4x4 & reg) {
    const float norm = float(xb->norm);
    const int qs_off = il * 4;
    float4x4 reg_f;

    for (int g = 0; g < 4; g++) {
        const uint8_t qb = xb->qs[qs_off + g];
        const uint8_t sb = xb->signs[il * 2 + g / 2];
        const int sshift = (g & 1) * 4;

        // Unpack 4 3-bit indices
        float4 raw;
        raw[0] = planar_centroids_3bit[(qb & 0x03)        | (((sb >> (sshift + 0)) & 1) << 2)];
        raw[1] = planar_centroids_3bit[((qb >> 2) & 0x03) | (((sb >> (sshift + 1)) & 1) << 2)];
        raw[2] = planar_centroids_3bit[((qb >> 4) & 0x03) | (((sb >> (sshift + 2)) & 1) << 2)];
        raw[3] = planar_centroids_3bit[((qb >> 6) & 0x03) | (((sb >> (sshift + 3)) & 1) << 2)];

        // Inverse Givens rotation on pairs
        // Element indices within block: il*16 + g*4 + {0,1,2,3}
        // Pair indices: (il*16 + g*4)/2 + {0,1}
        int base_elem = il * 16 + g * 4;
        int pair0 = base_elem / 2;
        int pair1 = pair0 + 1;

        float c0 = planar_cos_64[pair0];
        float s0 = planar_sin_64[pair0];
        float c1 = planar_cos_64[pair1];
        float s1 = planar_sin_64[pair1];

        // Pair 0: elements 0,1
        float f0 =  c0 * raw[0] + s0 * raw[1];
        float f1 = -s0 * raw[0] + c0 * raw[1];
        // Pair 1: elements 2,3
        float f2 =  c1 * raw[2] + s1 * raw[3];
        float f3 = -s1 * raw[2] + c1 * raw[3];

        reg_f[g] = float4(f0 * norm, f1 * norm, f2 * norm, f3 * norm);
    }
    reg = (type4x4) reg_f;
}

// Vec dequantize: 4 elements per call (il in {0..NL_PLANAR3_VEC-1})
template <typename type4>
void dequantize_planar3_0_t4(device const block_planar3_0 * xb, short il, thread type4 & reg) {
    const float norm = float(xb->norm);
    const uint8_t qb = xb->qs[il];
    const uint8_t sb = xb->signs[il / 2];
    const int sshift = (il & 1) * 4;

    float4 raw;
    raw[0] = planar_centroids_3bit[(qb & 0x03)        | (((sb >> (sshift + 0)) & 1) << 2)];
    raw[1] = planar_centroids_3bit[((qb >> 2) & 0x03) | (((sb >> (sshift + 1)) & 1) << 2)];
    raw[2] = planar_centroids_3bit[((qb >> 4) & 0x03) | (((sb >> (sshift + 2)) & 1) << 2)];
    raw[3] = planar_centroids_3bit[((qb >> 6) & 0x03) | (((sb >> (sshift + 3)) & 1) << 2)];

    // Inverse Givens rotation on pairs
    int base_elem = il * 4;
    int pair0 = base_elem / 2;
    int pair1 = pair0 + 1;

    float c0 = planar_cos_64[pair0];
    float s0 = planar_sin_64[pair0];
    float c1 = planar_cos_64[pair1];
    float s1 = planar_sin_64[pair1];

    float f0 =  c0 * raw[0] + s0 * raw[1];
    float f1 = -s0 * raw[0] + c0 * raw[1];
    float f2 =  c1 * raw[2] + s1 * raw[3];
    float f3 = -s1 * raw[2] + c1 * raw[3];

    reg = (type4) float4(f0 * norm, f1 * norm, f2 * norm, f3 * norm);
}


