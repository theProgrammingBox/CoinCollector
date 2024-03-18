#include "Noise.cuh"

struct Network {
    uint32_t* parameters;
    uint32_t layers;
    float learningRate;
    uint32_t batchSize;
    float weightDecay;
    float meanBeta;
    float varBeta;
    float meanCor;
    float varCor;
    float** outputs;
    float** outputGrad;
    float** weights;
    float** weightGrad;
    float** weightGradMean;
    float** weightGradVar;
};

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

__global__ void _fill(float* arr, uint32_t size, float value) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        arr[index] = value;
    }
}

void fill(float* arr, uint32_t size, float value) {
    _fill<<<(size >> 10) + (size & 0x3ff ? 1 : 0), 1024>>>(arr, size, value);
}

void initializeNetwork(Network* net, uint32_t* parameters, uint32_t layers, Noise* noise, float learningRate = 0.001f, uint32_t batchSize = 1, float weightDecay = 0, float meanBeta = 0.9, float varBeta = 0.999) {
    net->parameters = parameters;
    net->layers = layers;
    net->learningRate = learningRate;
    net->batchSize = batchSize;
    net->weightDecay = weightDecay;
    net->meanBeta = meanBeta;
    net->varBeta = varBeta;
    net->meanCor = 1.0f;
    net->varCor = 1.0f;
    net->outputs = (float**)malloc(sizeof(float*) * (layers + 1));
    net->outputGrad = (float**)malloc(sizeof(float*) * (layers + 1));
    net->weights = (float**)malloc(sizeof(float*) * layers);
    net->weightGrad = (float**)malloc(sizeof(float*) * layers);
    net->weightGradMean = (float**)malloc(sizeof(float*) * layers);
    net->weightGradVar = (float**)malloc(sizeof(float*) * layers);
    
    for (uint32_t i = 0; i < layers + 1; i++) {
        cudaMalloc(&net->outputs[i], sizeof(float) * net->batchSize * parameters[i]);
        cudaMalloc(&net->outputGrad[i], sizeof(float) * net->batchSize * parameters[i]);
    }
    
    for (uint32_t i = 0; i < layers; i++) {
        cudaMalloc(&net->weights[i], sizeof(float) * parameters[i] * parameters[i + 1]);
        cudaMalloc(&net->weightGrad[i], sizeof(float) * parameters[i] * parameters[i + 1]);
        cudaMalloc(&net->weightGradMean[i], sizeof(float) * parameters[i] * parameters[i + 1]);
        cudaMalloc(&net->weightGradVar[i], sizeof(float) * parameters[i] * parameters[i + 1]);
        
        // using relu initialization
        fillUniform(net->weights[i], parameters[i] * parameters[i + 1], noise, -1.0f / sqrtf(parameters[i]), 1.0f / sqrtf(parameters[i]));
        fill(net->weightGradMean[i], parameters[i] * parameters[i + 1], 0.0f);
        fill(net->weightGradVar[i], parameters[i] * parameters[i + 1], 0.0f);
        
        // printTensor(net->weights[i], parameters[i], parameters[i + 1]);
    }
}

__global__ void _reluForward(float *dTensor, uint32_t size) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    dTensor[idx] = dTensor[idx] > 0 ? dTensor[idx] : 0;
}

void reluForward(float *dTensor, uint32_t size) {
    _reluForward<<<(size >> 10) + (size & 0x3ff ? 1 : 0), 0x400>>>(dTensor, size);
}

void forwardPropagate(cublasHandle_t *cublasHandle, Network* net) {
    const float ONE = 1.0f;
    const float ZERO = 0.0f;
    
    for (uint32_t i = 0; i < net->layers - 1; i++) {
        cublasSgemm(*cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
            net->parameters[i + 1], net->batchSize, net->parameters[i],
            &ONE,
            net->weights[i], net->parameters[i + 1],
            net->outputs[i], net->parameters[i],
            &ZERO,
            net->outputs[i + 1], net->parameters[i + 1]);
            
        reluForward(net->outputs[i + 1], net->batchSize * net->parameters[i + 1]);
    }
    
    cublasSgemm(*cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
        net->parameters[net->layers], net->batchSize, net->parameters[net->layers - 1],
        &ONE,
        net->weights[net->layers - 1], net->parameters[net->layers],
        net->outputs[net->layers - 1], net->parameters[net->layers - 1],
        &ZERO,
        net->outputs[net->layers], net->parameters[net->layers]);
}

__global__ void _reluBackward(float *dTensor, float *dTensorGrad, uint32_t size) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    dTensorGrad[idx] = dTensor[idx] > 0 ? dTensorGrad[idx] : 0;
}

void reluBackward(float *dTensor, float *dTensorGrad, uint32_t size) {
    _reluBackward<<<(size >> 10) + (size & 0x3ff ? 1 : 0), 0x400>>>(dTensor, dTensorGrad, size);
}

__global__ void _integratedAdamUpdate(float *dTensor, float *dTensorGrad, float *dTensorMean, float *dTensorVar, uint32_t size, Network net) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float grad = dTensorGrad[idx];
    float mean = net.meanBeta * dTensorMean[idx] + (1.0f - net.meanBeta) * grad;
    float var = net.varBeta * dTensorVar[idx] + (1.0f - net.varBeta) * grad * grad;
    float meanCor = mean / (1.0f - net.meanCor);
    float varCor = var / (1.0f - net.varCor);
    dTensorMean[idx] = mean;
    dTensorVar[idx] = var;
    dTensor[idx] += net.learningRate * (meanCor / (sqrtf(varCor) + 1e-8f) - net.weightDecay * dTensor[idx]);
}

void integratedAdamUpdate(float *dTensor, float *dTensorGrad, float *dTensorMean, float *dTensorVar, uint32_t size, Network *net) {
    _integratedAdamUpdate<<<(size >> 10) + (size & 0x3ff ? 1 : 0), 0x400>>>(dTensor, dTensorGrad, dTensorMean, dTensorVar, size, *net);
}

void backwardPropagate(cublasHandle_t *cublasHandle, Network* net) {
    const float ONE = 1.0f;
    const float ZERO = 0.0f;
    
    net->meanCor *= net->meanBeta;
    net->varCor *= net->varBeta;
    
    cublasSgemm(*cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
        net->parameters[net->layers], net->parameters[net->layers - 1], net->batchSize,
        &ONE,
        net->outputGrad[net->layers], net->parameters[net->layers],
        net->outputs[net->layers - 1], net->parameters[net->layers - 1],
        &ZERO,
        net->weightGrad[net->layers - 1], net->parameters[net->layers]);
        
    cublasSgemm(*cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
        net->parameters[net->layers - 1], net->batchSize, net->parameters[net->layers],
        &ONE,
        net->weights[net->layers - 1], net->parameters[net->layers],
        net->outputGrad[net->layers], net->parameters[net->layers],
        &ZERO,
        net->outputGrad[net->layers - 1], net->parameters[net->layers - 1]);
                
    for (uint32_t i = net->layers - 1; i--;) {
        reluBackward(net->outputs[i + 1], net->outputGrad[i + 1], net->batchSize * net->parameters[i + 1]);
        cublasSgemm(*cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
            net->parameters[i + 1], net->parameters[i], net->batchSize,
            &ONE,
            net->outputGrad[i + 1], net->parameters[i + 1],
            net->outputs[i], net->parameters[i],
            &ZERO,
            net->weightGrad[i], net->parameters[i + 1]);
            
        cublasSgemm(*cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
            net->parameters[i], net->batchSize, net->parameters[i + 1],
            &ONE,
            net->weights[i], net->parameters[i + 1],
            net->outputGrad[i + 1], net->parameters[i + 1],
            &ZERO,
            net->outputGrad[i], net->parameters[i]);
    }
    
    for (uint32_t i = 0; i < net->layers; i++) {
        integratedAdamUpdate(net->weights[i], net->weightGrad[i], net->weightGradMean[i], net->weightGradVar[i], net->parameters[i] * net->parameters[i + 1], net);
    }
}

void copyParams(Network* net, Network* net2) {
    for (uint32_t i = 0; i < net->layers; i++) {
        cudaMemcpy(net2->weights[i], net->weights[i], net->parameters[i] * net->parameters[i + 1] * sizeof(float), cudaMemcpyDeviceToDevice);
    }
}

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

void printParams(Network* net) {
    for (uint32_t i = 0; i < net->layers; i++) {
        printf("Layer %d\n", i);
        printf("Output\n");
        printTensor(net->outputs[i], net->parameters[i], net->batchSize);
        printf("Weight\n");
        printTensor(net->weights[i], net->parameters[i + 1], net->parameters[i]);
    }
    printf("Output\n");
    printTensor(net->outputs[net->layers], net->parameters[net->layers], net->batchSize);
}

void printBackParams(Network* net) {
    for (uint32_t i = net->layers; i--;) {
        printf("Layer %d\n", i);
        printf("OutputGrad\n");
        printTensor(net->outputGrad[i], net->parameters[i + 1], net->batchSize);
        printf("WeightGrad\n");
        printTensor(net->weightGrad[i], net->parameters[i + 1], net->parameters[i]);
    }
    printf("OutputGrad\n");
    printTensor(net->outputGrad[0], net->parameters[1], net->batchSize);
}