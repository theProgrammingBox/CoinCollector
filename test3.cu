#include "Network.cuh"

int main(int argc, char** argv) {
    Noise noise;
    initializeNoise(&noise);
    
    Network net;
    uint32_t parameters[] = {3, 4, 1};
    uint32_t layers = sizeof(parameters) / sizeof(uint32_t) - 1;
    initializeNetwork(&net, parameters, layers, &noise, 0.1f, 1);
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    for (uint32_t i = 0; i < 100; i++) {
        float testInput[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        cudaMemcpy(net.outputs[0], testInput, net.batchSize * net.parameters[0] * sizeof(float), cudaMemcpyHostToDevice);
        forwardPropagate(&handle, &net);
        
        printTensor(net.outputs[net.layers], parameters[net.layers], net.batchSize);
        float output[net.batchSize * net.parameters[net.layers]];
        cudaMemcpy(output, net.outputs[net.layers], sizeof(output), cudaMemcpyDeviceToHost);
        
        float testOutputGrad[2 * 1];
        for (uint32_t i = 0; i < 2; i++) {
            testOutputGrad[i] = 1 - output[i];
        }
        cudaMemcpy(net.outputGrad[net.layers], testOutputGrad, net.batchSize * net.parameters[net.layers] * sizeof(float), cudaMemcpyHostToDevice);
        backwardPropagate(&handle, &net);
    }
    
    printParams(&net);
    
    return 0;
}