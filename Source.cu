#include <stdio.h>
#include <stdint.h>

#include "Header.cuh"

int main() {
    printf("Hello World!\n");
    
    uint64_t size = 0x800;
    uint8_t *d_data;
    checkCudaStatus(cudaMalloc(&d_data, size));
    test<<<size >> 10, 0x400>>>(d_data, size);
    printDeviceTensor(d_data, size);
    
    return 0;
}