#include <cuda_runtime.h>

inline void checkCudaStatus(cudaError_t status) {
    if (status != cudaSuccess) {
        printf("cuda API failed with status %d: %s\n", status, cudaGetErrorString(status));
        exit(-1);
    }
}

__device__ int fastFloor(float x) {
    int xi = (int) x;
    return x < xi ? xi - 1 : xi;
}

__global__ void test(uint8_t *data, uint32_t size) {
    const float STRETCH_CONSTANT_4D = -0.138196601125011;
    const float SQUISH_CONSTANT_4D = 0.309016994374947;
    const float NORM_CONSTANT_4D = 30.0;
    const signed char gradients4D[] = {
        3,  1,  1,  1,      1,  3,  1,  1,      1,  1,  3,  1,      1,  1,  1,  3,
        -3,  1,  1,  1,     -1,  3,  1,  1,     -1,  1,  3,  1,     -1,  1,  1,  3,
        3, -1,  1,  1,      1, -3,  1,  1,      1, -1,  3,  1,      1, -1,  1,  3,
        -3, -1,  1,  1,     -1, -3,  1,  1,     -1, -1,  3,  1,     -1, -1,  1,  3,
        3,  1, -1,  1,      1,  3, -1,  1,      1,  1, -3,  1,      1,  1, -1,  3,
        -3,  1, -1,  1,     -1,  3, -1,  1,     -1,  1, -3,  1,     -1,  1, -1,  3,
        3, -1, -1,  1,      1, -3, -1,  1,      1, -1, -3,  1,      1, -1, -1,  3,
        -3, -1, -1,  1,     -1, -3, -1,  1,     -1, -1, -3,  1,     -1, -1, -1,  3,
        3,  1,  1, -1,      1,  3,  1, -1,      1,  1,  3, -1,      1,  1,  1, -3,
        -3,  1,  1, -1,     -1,  3,  1, -1,     -1,  1,  3, -1,     -1,  1,  1, -3,
        3, -1,  1, -1,      1, -3,  1, -1,      1, -1,  3, -1,      1, -1,  1, -3,
        -3, -1,  1, -1,     -1, -3,  1, -1,     -1, -1,  3, -1,     -1, -1,  1, -3,
        3,  1, -1, -1,      1,  3, -1, -1,      1,  1, -3, -1,      1,  1, -1, -3,
        -3,  1, -1, -1,     -1,  3, -1, -1,     -1,  1, -3, -1,     -1,  1, -1, -3,
        3, -1, -1, -1,      1, -3, -1, -1,      1, -1, -3, -1,      1, -1, -1, -3,
        -3, -1, -1, -1,     -1, -3, -1, -1,     -1, -1, -3, -1,     -1, -1, -1, -3,
    };
    
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    float cos, sin;
    sincosf((float)idx, &sin, &cos);
    data[idx] = fastFloor((sin * sin + cos * cos) * 10.1);
}

// print device tensor by copying it to host memory
void printDeviceTensor(uint8_t *d_data, uint32_t size) {
    uint8_t *h_data = (uint8_t *) malloc(size);
    checkCudaStatus(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));
    for (uint32_t i = 0; i < size; i++) {
        printf("%d ", h_data[i]);
    }
    printf("\n");
    free(h_data);
}