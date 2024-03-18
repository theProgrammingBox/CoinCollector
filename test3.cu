#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/time.h>

#include <cublas_v2.h>

#define BATCH_SIZE 2
#define LEARNING_RATE 0.1

#define MEAN_BETA 0.9
#define VAR_BETA 0.999
#define WEIGHT_DECAY 0.001

void printTensor(float* tensor, uint32_t width, uint32_t height) {
    float* arr = (float*)malloc(height * width * sizeof(float));
    cudaMemcpy(arr, tensor, height * width * sizeof(float), cudaMemcpyDeviceToHost);
    for (uint32_t i = 0; i < height; i++) {
        for (uint32_t j = 0; j < width; j++) {
            printf("%f ", arr[i * width + j]);
        }
        printf("\n");
    }
    printf("\n");
    free(arr);
}

__global__ void _reluForward(float *dTensor, uint32_t size) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    dTensor[idx] = dTensor[idx] > 0 ? dTensor[idx] : 0;
}

void reluForward(float *dTensor, uint32_t size) {
    _reluForward<<<(size >> 10) + (size & 0x3ff ? 1 : 0), 0x400>>>(dTensor, size);
}

__global__ void _reluBackward(float *dTensor, float *dTensorGrad, uint32_t size) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    dTensorGrad[idx] = dTensor[idx] > 0 ? dTensorGrad[idx] : 0;
}

void reluBackward(float *dTensor, float *dTensorGrad, uint32_t size) {
    _reluBackward<<<(size >> 10) + (size & 0x3ff ? 1 : 0), 0x400>>>(dTensor, dTensorGrad, size);
}

__global__ void _integratedAdamUpdate(float *dTensor, float *dTensorGrad, float *dTensorMean, float *dTensorVar, float betaMeanCor, float betaVarCor, uint32_t size) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float grad = dTensorGrad[idx];
    float mean = MEAN_BETA * dTensorMean[idx] + (1.0f - MEAN_BETA) * grad;
    float var = VAR_BETA * dTensorVar[idx] + (1.0f - VAR_BETA) * grad * grad;
    float meanCor = mean / (1.0f - betaMeanCor);
    float varCor = var / (1.0f - betaVarCor);
    dTensorMean[idx] = mean;
    dTensorVar[idx] = var;
    dTensor[idx] += LEARNING_RATE * (meanCor / (sqrtf(varCor) + 1e-8f) - WEIGHT_DECAY * dTensor[idx]);
}

void integratedAdamUpdate(float *dTensor, float *dTensorGrad, float *dTensorMean, float *dTensorVar, float betaMeanCor, float betaVarCor, uint32_t size) {
    _integratedAdamUpdate<<<(size >> 10) + (size & 0x3ff ? 1 : 0), 0x400>>>(dTensor, dTensorGrad, dTensorMean, dTensorVar, betaMeanCor, betaVarCor, size);
}

struct Noise {
    uint32_t seed1;
    uint32_t seed2;
};

uint32_t genNoise(Noise* noise) {
    noise->seed2 ^= noise->seed1 >> 17;
    noise->seed2 *= 0xbf324c81;
    noise->seed1 ^= noise->seed2 >> 13;
    noise->seed1 *= 0x9c7493ad;
    return noise->seed1;
}

void initializeNoise(Noise* noise) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    noise->seed1 = tv.tv_sec;
    noise->seed2 = tv.tv_usec;
    for (uint8_t i = 4; i--;) genNoise(noise);
}

__global__ void _fillUniform(float* arr, uint32_t size, Noise noise, float lowerBound, float upperBound) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        uint32_t hash = (index ^ noise.seed1) * 0x4ba1bb47;
        hash ^= (hash >> 17);
        hash ^= (hash ^ noise.seed2) * 0xb7ebcb79;
        hash ^= (hash >> 13);
        arr[index] = (float)hash / 0xffffffff * (upperBound - lowerBound) + lowerBound;
    }
}

void fillUniform(float* arr, uint32_t size, Noise* noise, float lowerBound, float upperBound) {
    genNoise(noise);
    _fillUniform<<<(size >> 10) + (size & 0x3ff ? 1 : 0), 1024>>>(arr, size, *noise, lowerBound, upperBound);
}

__global__ void _fill(float* arr, uint32_t size, Noise noise, float value) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        arr[index] = value;
    }
}

void fill(float* arr, uint32_t size, Noise* noise, float value) {
    _fill<<<(size >> 10) + (size & 0x3ff ? 1 : 0), 1024>>>(arr, size, *noise, value);
}

struct Network {
  uint32_t* parameters;
  uint32_t layers;
  float meanCor;
  float VarCor;
  float** outputs;
  float** outputGrad;
  float** weights;
  float** weightGrad;
  float** weightGradMean;
  float** weightGradVar;
};

void initializeNetwork(Network* net, uint32_t* parameters, uint32_t layers, Noise* noise) {
    net->parameters = parameters;
    net->layers = layers;
    net->meanCor = 1.0f;
    net->VarCor = 1.0f;
    net->outputs = (float**)malloc(sizeof(float*) * (layers + 1));
    net->outputGrad = (float**)malloc(sizeof(float*) * (layers + 1));
    net->weights = (float**)malloc(sizeof(float*) * layers);
    net->weightGrad = (float**)malloc(sizeof(float*) * layers);
    net->weightGradMean = (float**)malloc(sizeof(float*) * layers);
    net->weightGradVar = (float**)malloc(sizeof(float*) * layers);
    
    for (uint32_t i = 0; i < layers + 1; i++) {
        cudaMalloc(&net->outputs[i], sizeof(float) * BATCH_SIZE * parameters[i]);
        cudaMalloc(&net->outputGrad[i], sizeof(float) * BATCH_SIZE * parameters[i]);
    }
    
    for (uint32_t i = 0; i < layers; i++) {
        cudaMalloc(&net->weights[i], sizeof(float) * parameters[i] * parameters[i + 1]);
        cudaMalloc(&net->weightGrad[i], sizeof(float) * parameters[i] * parameters[i + 1]);
        cudaMalloc(&net->weightGradMean[i], sizeof(float) * parameters[i] * parameters[i + 1]);
        cudaMalloc(&net->weightGradVar[i], sizeof(float) * parameters[i] * parameters[i + 1]);
        
        // using relu initialization
        fillUniform(net->weights[i], parameters[i] * parameters[i + 1], noise, -1.0f / sqrtf(parameters[i]), 1.0f / sqrtf(parameters[i]));
        fill(net->weightGradMean[i], parameters[i] * parameters[i + 1], noise, 0.0f);
        fill(net->weightGradVar[i], parameters[i] * parameters[i + 1], noise, 0.0f);
        
        // printTensor(net->weights[i], parameters[i], parameters[i + 1]);
    }
}

void forwardPropagate(cublasHandle_t *cublasHandle, Network* net) {
    const float ONE = 1.0f;
    const float ZERO = 0.0f;
    
    for (uint32_t i = 0; i < net->layers - 1; i++) {
        cublasSgemm(*cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
            net->parameters[i + 1], BATCH_SIZE, net->parameters[i],
            &ONE,
            net->weights[i], net->parameters[i + 1],
            net->outputs[i], net->parameters[i],
            &ZERO,
            net->outputs[i + 1], net->parameters[i + 1]);
            
        // printf("input\n");
        // printTensor(net->outputs[i], net->parameters[i], BATCH_SIZE);
        // printf("weight\n");
        // printTensor(net->weights[i], net->parameters[i + 1], net->parameters[i]);
        // printf("output\n");
        // printTensor(net->outputs[i + 1], net->parameters[i + 1], BATCH_SIZE);
        reluForward(net->outputs[i + 1], BATCH_SIZE * net->parameters[i + 1]);
        // printf("output after relu\n");
        // printTensor(net->outputs[i + 1], net->parameters[i + 1], BATCH_SIZE);
    }
    
    cublasSgemm(*cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
        net->parameters[net->layers], BATCH_SIZE, net->parameters[net->layers - 1],
        &ONE,
        net->weights[net->layers - 1], net->parameters[net->layers],
        net->outputs[net->layers - 1], net->parameters[net->layers - 1],
        &ZERO,
        net->outputs[net->layers], net->parameters[net->layers]);
        
    // printf("weight\n");
    // printTensor(net->weights[net->layers - 1], net->parameters[net->layers], net->parameters[net->layers - 1]);
    // printf("output\n");
    // printTensor(net->outputs[net->layers], net->parameters[net->layers], BATCH_SIZE);
}

void backwardPropagate(cublasHandle_t *cublasHandle, Network* net) {
    const float ONE = 1.0f;
    const float ZERO = 0.0f;
    
    net->meanCor *= MEAN_BETA;
    net->VarCor *= VAR_BETA;
    
    cublasSgemm(*cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
        net->parameters[net->layers], net->parameters[net->layers - 1], BATCH_SIZE,
        &ONE,
        net->outputGrad[net->layers], net->parameters[net->layers],
        net->outputs[net->layers - 1], net->parameters[net->layers - 1],
        &ZERO,
        net->weightGrad[net->layers - 1], net->parameters[net->layers]);
        
    cublasSgemm(*cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
        net->parameters[net->layers - 1], BATCH_SIZE, net->parameters[net->layers],
        &ONE,
        net->weights[net->layers - 1], net->parameters[net->layers],
        net->outputGrad[net->layers], net->parameters[net->layers],
        &ZERO,
        net->outputGrad[net->layers - 1], net->parameters[net->layers - 1]);
                
    // printf("prev output\n");
    // printTensor(net->outputs[net->layers - 1], net->parameters[net->layers - 1], BATCH_SIZE);
    // printf("output grad\n");
    // printTensor(net->outputGrad[net->layers], net->parameters[net->layers], BATCH_SIZE);
    // printf("weight grad\n\n");
    // printTensor(net->weightGrad[net->layers - 1], net->parameters[net->layers], net->parameters[net->layers - 1]);
    // printf("output grad\n");
    // printTensor(net->outputGrad[net->layers], net->parameters[net->layers], BATCH_SIZE);
    // printf("weight\n");
    // printTensor(net->weights[net->layers - 1], net->parameters[net->layers], net->parameters[net->layers - 1]);
    // printf("prev output grad\n\n\n");
    // printTensor(net->outputGrad[net->layers - 1], net->parameters[net->layers - 1], BATCH_SIZE);
    
    for (uint32_t i = net->layers - 1; i--;) {
        reluBackward(net->outputs[i + 1], net->outputGrad[i + 1], BATCH_SIZE * net->parameters[i + 1]);
        
        // printf("output grad\n");
        // printTensor(net->outputGrad[i + 1], net->parameters[i + 1], BATCH_SIZE);
        
        cublasSgemm(*cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
            net->parameters[i + 1], net->parameters[i], BATCH_SIZE,
            &ONE,
            net->outputGrad[i + 1], net->parameters[i + 1],
            net->outputs[i], net->parameters[i],
            &ZERO,
            net->weightGrad[i], net->parameters[i + 1]);
            
        cublasSgemm(*cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
            net->parameters[i], BATCH_SIZE, net->parameters[i + 1],
            &ONE,
            net->weights[i], net->parameters[i + 1],
            net->outputGrad[i + 1], net->parameters[i + 1],
            &ZERO,
            net->outputGrad[i], net->parameters[i]);
            
        // printf("prev output\n");
        // printTensor(net->outputs[i], net->parameters[i], BATCH_SIZE);
        // printf("output grad\n");
        // printTensor(net->outputGrad[i + 1], net->parameters[i + 1], BATCH_SIZE);
        // printf("weight grad\n\n");
        // printTensor(net->weightGrad[i], net->parameters[i + 1], net->parameters[i]);
        // printf("output grad\n");
        // printTensor(net->outputGrad[i + 1], net->parameters[i + 1], BATCH_SIZE);
        // printf("weight\n");
        // printTensor(net->weights[i], net->parameters[i + 1], net->parameters[i]);
        // printf("prev output grad\n\n\n");
        // printTensor(net->outputGrad[i], net->parameters[i], BATCH_SIZE);
    }
    
    for (uint32_t i = 0; i < net->layers; i++) {
        integratedAdamUpdate(net->weights[i], net->weightGrad[i], net->weightGradMean[i], net->weightGradVar[i], net->meanCor, net->VarCor, net->parameters[i] * net->parameters[i + 1]);
    }
}

void printParams(Network* net) {
    for (uint32_t i = 0; i < net->layers; i++) {
        printf("Layer %d\n", i);
        printf("Output\n");
        printTensor(net->outputs[i], net->parameters[i], BATCH_SIZE);
        printf("Weight\n");
        printTensor(net->weights[i], net->parameters[i + 1], net->parameters[i]);
    }
    printf("Output\n");
    printTensor(net->outputs[net->layers], net->parameters[net->layers], BATCH_SIZE);
}

int main(int argc, char** argv) {
    Noise noise;
    initializeNoise(&noise);
    
    Network net;
    uint32_t parameters[] = {3, 4, 1};
    uint32_t layers = sizeof(parameters) / sizeof(uint32_t) - 1;
    initializeNetwork(&net, parameters, layers, &noise);
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    for (uint32_t i = 0; i < 100; i++) {
        float testInput[2 * 3] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        cudaMemcpy(net.outputs[0], testInput, sizeof(testInput), cudaMemcpyHostToDevice);
        forwardPropagate(&handle, &net);
        
        printTensor(net.outputs[net.layers], parameters[net.layers], BATCH_SIZE);
        float output[2 * 1];
        cudaMemcpy(output, net.outputs[net.layers], sizeof(output), cudaMemcpyDeviceToHost);
        
        float testOutputGrad[2 * 1];
        for (uint32_t i = 0; i < 2; i++) {
            testOutputGrad[i] = 1 - output[i];
        }
        cudaMemcpy(net.outputGrad[net.layers], testOutputGrad, sizeof(testOutputGrad), cudaMemcpyHostToDevice);
        backwardPropagate(&handle, &net);
    }
    
    printParams(&net);
    
    return 0;
}