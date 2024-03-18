#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/time.h>

#include <cublas_v2.h>

#define BATCH_SIZE 64

void printTensor(float* tensor, uint32_t rows, uint32_t cols) {
    float* arr = (float*)malloc(rows * cols * sizeof(float));
    cudaMemcpy(arr, tensor, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
    for (uint32_t i = 0; i < rows; i++) {
        for (uint32_t j = 0; j < cols; j++) {
            printf("%f ", arr[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
    free(arr);
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
  uint8_t* biases;
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

void initializeNetwork(Network* net, uint32_t* parameters, uint32_t* biases, uint32_t layers, Noise* noise) {
    net->parameters = parameters;
    net->biases = biases;
    net->layers = layers;
    net->meanCor = 1.0f;
    net->VarCor = 1.0f;
    net->outputs = (float**)malloc(sizeof(float*) * (layers + 2));
    net->outputGrad = (float**)malloc(sizeof(float*) * (layers + 2));
    net->weights = (float**)malloc(sizeof(float*) * (layers + 1));
    net->weightGrad = (float**)malloc(sizeof(float*) * (layers + 1));
    net->weightGradMean = (float**)malloc(sizeof(float*) * (layers + 1));
    net->weightGradVar = (float**)malloc(sizeof(float*) * (layers + 1));
    
    for (uint32_t i = 0; i < layers + 2; i++) {
        cudaMalloc(&net->outputs[i], sizeof(float) * BATCH_SIZE * parameters[i]);
        cudaMalloc(&net->outputGrad[i], sizeof(float) * BATCH_SIZE * parameters[i]);
    }
    
    for (uint32_t i = 0; i < layers + 1; i++) {
        cudaMalloc(&net->weights[i], sizeof(float) * parameters[i] * parameters[i + 1]);
        cudaMalloc(&net->weightGrad[i], sizeof(float) * parameters[i] * parameters[i + 1]);
        cudaMalloc(&net->weightGradMean[i], sizeof(float) * parameters[i] * parameters[i + 1]);
        cudaMalloc(&net->weightGradVar[i], sizeof(float) * parameters[i] * parameters[i + 1]);
        
        fillUniform(net->weights[i], parameters[i] * parameters[i + 1], noise, -1.0f, 1.0f);
        fill(net->weightGradMean[i], parameters[i] * parameters[i + 1], noise, 0.0f);
        fill(net->weightGradVar[i], parameters[i] * parameters[i + 1], noise, 0.0f);
        
        printTensor(net->weights[i], parameters[i], parameters[i + 1]);
    }
}

int main(int argc, char** argv) {
    Noise noise;
    initializeNoise(&noise);
    
    Network net;
    uint32_t parameters[] = {4, 16, 4};
    uint8_t biases[] = {0, 1, 1};
    uint32_t layers = sizeof(parameters) / sizeof(uint32_t) - 2;
    initializeNetwork(&net, parameters, biases, layers, &noise);
    
    return 0;
}