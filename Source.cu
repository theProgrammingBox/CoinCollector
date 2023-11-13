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

void printDeviceTensor16(const uint16_t *d_data, uint32_t size) {
    uint16_t *h_data = (uint16_t *) malloc(size * sizeof(uint16_t));
    checkCudaStatus(cudaMemcpy(h_data, d_data, size * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    for (uint32_t i = 0; i < size; i++) {
        printf("%d ", h_data[i]);
    }
    printf("\n\n");
    free(h_data);
}

void printDeviceTensor8(const uint8_t *d_data, uint32_t size) {
    uint8_t *h_data = (uint8_t *) malloc(size * sizeof(uint8_t));
    checkCudaStatus(cudaMemcpy(h_data, d_data, size * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    for (uint32_t i = 0; i < size; i++) {
        printf("%d ", (int8_t)h_data[i]);
    }
    printf("\n\n");
    free(h_data);
}

int main() {
    uint32_t seed = time(NULL);
    for (uint8_t i = 16; i--;) rand32(&seed);
    
    uint16_t permSize = 256;
    uint64_t size = 0x100000000;
    
    uint16_t *d_perm;
    uint8_t *d_data;
    
    checkCudaStatus(cudaMalloc((void**)&d_perm, permSize * sizeof(uint16_t)));
    checkCudaStatus(cudaMalloc((void**)&d_data, size * sizeof(uint8_t)));
    
    fillTensor<<<1, 256>>>(d_perm, seed);
    test<<<size >> 10, 1024>>>(d_data, d_perm);
    
    printDeviceTensor16(d_perm, 256);
    printDeviceTensor8(d_data, 256);
    
    checkCudaStatus(cudaFree(d_data));
    checkCudaStatus(cudaFree(d_perm));
    return 0;
}