#include <stdio.h>
#include <stdint.h>
#include <time.h>

#include "Header.cuh"

uint32_t rand32(uint32_t* seed) {
    *seed ^= *seed >> 16;
    *seed *= 0xBAC57D37;
    *seed ^= *seed >> 16;
    *seed *= 0x24F66AC9;
    return *seed;
}

int main() {
    uint32_t seed = time(NULL);
    for (uint8_t i = 16; i--;) rand32(&seed);
    
    uint16_t permSize = 256;
    uint64_t size = 1024;
    
    uint16_t *d_perm;
    uint8_t *d_data;
    
    checkCudaStatus(cudaMalloc((void**)&d_perm, permSize * sizeof(uint16_t)));
    checkCudaStatus(cudaMalloc((void**)&d_data, size * sizeof(uint8_t)));
    
    fillTensor<<<1, permSize>>>(d_perm, seed);
    test<<<size >> 10, 1024>>>(d_data, d_perm, size);
    
    printDeviceTensor16(d_perm, permSize);
    printDeviceTensor8(d_data, size);
    
    checkCudaStatus(cudaFree(d_data));
    checkCudaStatus(cudaFree(d_perm));
    return 0;
}