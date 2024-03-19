#include "Network.cuh"

// error handling for cudaMalloc
void checkMallocError(cudaError_t err) {
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
        exit(1);
    }
}

int main(int argc, char** argv) {
    Noise noise;
    initNoise(&noise);
    
    Network net;
    uint32_t parameters[] = {3, 4, 1};
    uint32_t layers = sizeof(parameters) / sizeof(uint32_t) - 1;
    initNetwork(&net, parameters, layers, &noise, 0.1f, 1);
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    float* testArr;
    // cudaMalloc((void**)&testArr, 3 * sizeof(float));
    checkMallocError(cudaMalloc((void**)&testArr, 3 * sizeof(float)));
    float testArrHost[] = {1.0f, 2.0f, 3.0f};
    cudaMemcpy(testArr, testArrHost, 3 * sizeof(float), cudaMemcpyHostToDevice);
    printf("Test array:\n");
    printTensor(testArr, 3, 1);
    
    for (uint32_t i = 0; i < 0; i++) {
        float testInput[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        cudaMemcpy(net.outputs[0], testInput, net.batchSize * net.parameters[0] * sizeof(float), cudaMemcpyHostToDevice);
        printf("Input:\n");
        printTensor(net.outputs[0], parameters[0], net.batchSize);
        // forwardNoiseless(&handle, &net);
        
        // printTensor(net.outputs[net.layers], parameters[net.layers], net.batchSize);
        // float output[net.batchSize * net.parameters[net.layers]];
        // cudaMemcpy(output, net.outputs[net.layers], sizeof(output), cudaMemcpyDeviceToHost);
        
        // float testOutputGrad[2 * 1];
        // for (uint32_t i = 0; i < 2; i++) {
        //     testOutputGrad[i] = 1 - output[i];
        // }
        // cudaMemcpy(net.outputGrads[net.layers], testOutputGrad, net.batchSize * net.parameters[net.layers] * sizeof(float), cudaMemcpyHostToDevice);
        // backwardNoiseless(&handle, &net);
    }
    
    printParams(&net);
    
    return 0;
}