#include <stdint.h>

inline void checkCudaStatus(cudaError_t status) {
    if (status != cudaSuccess) {
        printf("cuda API failed with status %d: %s\n", status, cudaGetErrorString(status));
        exit(-1);
    }
}

__global__ void fillDPerm(uint16_t *perm, uint32_t seed) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	seed ^= idx;
    seed ^= seed << 13;
    seed *= 0xBAC57D37;
    seed ^= seed >> 17;
    seed *= 0x24F66AC9;
    seed ^= seed << 5;
    perm[idx] = seed & 0x1F;
}

__device__ float grad4(uint8_t hash, float x, float y, float z, float w) {
    float u = hash < 24 ? x : y;
    float v = hash < 16 ? y : z;
    float s = hash < 8 ? z : w;
    return (hash & 1 ? -u : u) + (hash & 2 ? -v : v) + (hash & 4 ? -s : s);
}

#define norm16 0.00009587379924285
#define F4 0.309016994
#define G4 0.138196601
#define FASTFLOOR(x) ( ((int)(x)<=(x)) ? ((int)x) : (((int)x)-1) )

__global__ void fillDData(uint8_t *dData, const uint16_t *perm) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    float cosx, sinx, cosy, siny;
    sincosf((idx & 0xFFFF) * norm16, &sinx, &cosx);
    sincosf((idx >> 16) * norm16, &siny, &cosy);
    
    float x = cosx * 800;
    float y = sinx * 800;
    float z = cosy * 800;
    float w = siny * 800;
    
    float n0, n1, n2, n3, n4;
    
    float s = (x + y + z + w) * F4;
    float xs = x + s;
    float ys = y + s;
    float zs = z + s;
    float ws = w + s;
    
    int32_t i = FASTFLOOR(xs);
    int32_t j = FASTFLOOR(ys);
    int32_t k = FASTFLOOR(zs);
    int32_t l = FASTFLOOR(ws);
    
    float t = (i + j + k + l) * G4;
    float X0 = i - t;
    float Y0 = j - t;
    float Z0 = k - t;
    float W0 = l - t;
    
    float x0 = x - X0;
    float y0 = y - Y0;
    float z0 = z - Z0;
    float w0 = w - W0;
    
    uint8_t c = uint8_t(x0 > y0) << 5 | uint8_t(x0 > z0) << 4 | uint8_t(y0 > z0) << 3 | uint8_t(x0 > w0) << 2 | uint8_t(y0 > w0) << 1 | uint8_t(z0 > w0);
    
    uint8_t i1, j1, k1, l1;
    uint8_t i2, j2, k2, l2;
    uint8_t i3, j3, k3, l3;
    
    static unsigned char simplex[64][4] = {
    {0,1,2,3},{0,1,3,2},{0,0,0,0},{0,2,3,1},{0,0,0,0},{0,0,0,0},{0,0,0,0},{1,2,3,0},
    {0,2,1,3},{0,0,0,0},{0,3,1,2},{0,3,2,1},{0,0,0,0},{0,0,0,0},{0,0,0,0},{1,3,2,0},
    {0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},
    {1,2,0,3},{0,0,0,0},{1,3,0,2},{0,0,0,0},{0,0,0,0},{0,0,0,0},{2,3,0,1},{2,3,1,0},
    {1,0,2,3},{1,0,3,2},{0,0,0,0},{0,0,0,0},{0,0,0,0},{2,0,3,1},{0,0,0,0},{2,1,3,0},
    {0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},
    {2,0,1,3},{0,0,0,0},{0,0,0,0},{0,0,0,0},{3,0,1,2},{3,0,2,1},{0,0,0,0},{3,1,2,0},
    {2,1,0,3},{0,0,0,0},{0,0,0,0},{0,0,0,0},{3,1,0,2},{0,0,0,0},{3,2,0,1},{3,2,1,0}};
    
    i1 = simplex[c][0]>=3;
    j1 = simplex[c][1]>=3;
    k1 = simplex[c][2]>=3;
    l1 = simplex[c][3]>=3;
    
    i2 = simplex[c][0]>=2;
    j2 = simplex[c][1]>=2;
    k2 = simplex[c][2]>=2;
    l2 = simplex[c][3]>=2;
    
    i3 = simplex[c][0]>=1;
    j3 = simplex[c][1]>=1;
    k3 = simplex[c][2]>=1;
    l3 = simplex[c][3]>=1;
    
    float x1 = x0 - i1 + G4;
    float y1 = y0 - j1 + G4;
    float z1 = z0 - k1 + G4;
    float w1 = w0 - l1 + G4;
    
    float x2 = x0 - i2 + 2.0f * G4;
    float y2 = y0 - j2 + 2.0f * G4;
    float z2 = z0 - k2 + 2.0f * G4;
    float w2 = w0 - l2 + 2.0f * G4;
        
    float x3 = x0 - i3 + 3.0f * G4;
    float y3 = y0 - j3 + 3.0f * G4;
    float z3 = z0 - k3 + 3.0f * G4;
    float w3 = w0 - l3 + 3.0f * G4;
    
    float x4 = x0 - 1.0f + 4.0f * G4;
    float y4 = y0 - 1.0f + 4.0f * G4;
    float z4 = z0 - 1.0f + 4.0f * G4;
    float w4 = w0 - 1.0f + 4.0f * G4;
    
    uint8_t ii = i;
    uint8_t jj = j;
    uint8_t kk = k;
    uint8_t ll = l;
    
    float t0 = 0.5f - x0*x0 - y0*y0 - z0*z0 - w0*w0;
    if (t0 < 0.0f) n0 = 0.0f;
    else {
        t0 *= t0;
        n0 = t0 * t0 * grad4(perm[ii + perm[jj + perm[kk + perm[ll]]]], x0, y0, z0, w0);
    }
    
    float t1 = 0.5f - x1*x1 - y1*y1 - z1*z1 - w1*w1;
    if (t1 < 0.0f) n1 = 0.0f;
    else {
        t1 *= t1;
        n1 = t1 * t1 * grad4(perm[ii + i1 + perm[jj + j1 + perm[kk + k1 + perm[ll + l1]]]], x1, y1, z1, w1);
    }
    
    float t2 = 0.5f - x2*x2 - y2*y2 - z2*z2 - w2*w2;
    if (t2 < 0.0f) n2 = 0.0f;
    else {
        t2 *= t2;
        n2 = t2 * t2 * grad4(perm[ii + i2 + perm[jj + j2 + perm[kk + k2 + perm[ll + l2]]]], x2, y2, z2, w2);
    }
    
    float t3 = 0.5f - x3*x3 - y3*y3 - z3*z3 - w3*w3;
    if (t3 < 0.0f) n3 = 0.0f;
    else {
        t3 *= t3;
        n3 = t3 * t3 * grad4(perm[ii + i3 + perm[jj + j3 + perm[kk + k3 + perm[ll + l3]]]], x3, y3, z3, w3);
    }
    
    float t4 = 0.5f - x4*x4 - y4*y4 - z4*z4 - w4*w4;
    if (t4 < 0.0f) n4 = 0.0f;
    else {
        t4 *= t4;
        n4 = t4 * t4 * grad4(perm[ii + 1 + perm[jj + 1 + perm[kk + 1 + perm[ll + 1]]]], x4, y4, z4, w4);
    }
    
    dData[idx] = (62.0f * (n0 + n1 + n2 + n3 + n4) + 1.0f) * 4;
}