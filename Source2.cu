#include <stdio.h>
#include <stdint.h>
#include <sys/time.h>

#include "Header2.cuh"

void printD16(const uint16_t *dData, uint32_t size) {
    uint16_t *hData = (uint16_t*)malloc(size * sizeof(uint16_t));
    checkCudaStatus(cudaMemcpy(hData, dData, size * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    for (uint32_t i = 0; i < size; i++) printf("%d ", hData[i]);
    printf("\n\n");
    free(hData);
}

void printD8(const uint8_t *dData, uint32_t size) {
    uint8_t *hData = (uint8_t*)malloc(size * sizeof(uint8_t));
    checkCudaStatus(cudaMemcpy(hData, dData, size * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    for (uint32_t i = 0; i < size; i++) printf("%d ", hData[i]);
    printf("\n\n");
    free(hData);
}

int main() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    uint32_t seed = tv.tv_sec ^ tv.tv_usec ^ 0x85f35457;
    for (int i = 8; i--;) {
        seed *= 0xBAC57D37;
        seed ^= seed << 13;
        seed *= 0x24F66AC9;
        seed ^= seed >> 17;
    }
    
    uint16_t *dPerm;
    uint8_t *dData;
    
    checkCudaStatus(cudaMalloc((void**)&dPerm, 0x100 * sizeof(uint16_t)));
    checkCudaStatus(cudaMalloc((void**)&dData, 0x100000000 * sizeof(uint8_t)));
    
    fillDPerm<<<1, 0x100>>>(dPerm, seed);
    fillDData<<<0x400000, 0x400>>>(dData, dPerm);
    
    printD16(dPerm, 0x100);
    printD8(dData, 0x100);
    
    checkCudaStatus(cudaFree(dPerm));
    checkCudaStatus(cudaFree(dData));
    
    printf("Seed: %u\n", seed);
    return 0;
}