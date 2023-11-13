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
    perm[idx] = seed;
}

#define norm16 0.00009587379924285

__global__ void fillDData(uint8_t *dData, const uint16_t *perm) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    float cosx, sinx, cosy, siny;
    sincosf((idx & 0xFFFF) * norm16, &sinx, &cosx);
    sincosf((idx >> 16) * norm16, &siny, &cosy);
    dData[idx] = (cosx * cosy + sinx * siny) * 127.5 + 127.5;
}