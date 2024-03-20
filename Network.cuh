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
    float** outputGrads;
    
    float** noisyWeights;
    float** weightMeans;
    float** weightVars;
    float** weightSamples;
    
    float** weightGrads;
    
    float** weightMeanGradMeans;
    float** weightVarGradMeans;
    
    float** weightMeanGradVars;
    float** weightVarGradVars;
};

__global__ void _fillUniform(float* arr, uint32_t size, Noise noise, float lowerBound, float upperBound) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        uint32_t hash = (index ^ noise.seed1) * 0x4ba1bb47;
        hash ^= (hash >> 17);
        hash ^= (hash ^ noise.seed2) * 0xb7ebcb79;
        hash ^= (hash << 13);
        arr[index] = (float)hash / 0xffffffff * (upperBound - lowerBound) + lowerBound;
    }
}

void fillUniform(float* arr, uint32_t size, Noise* noise, float lowerBound, float upperBound) {
    mix(noise);
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

void initNetwork(Network* net, uint32_t* parameters, uint32_t layers, Noise* noise, float learningRate = 0.001f, uint32_t batchSize = 1, float weightDecay = 0, float meanBeta = 0.9, float varBeta = 0.999) {
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
    net->outputGrads = (float**)malloc(sizeof(float*) * (layers + 1));
    
    net->noisyWeights = (float**)malloc(sizeof(float*) * layers);
    net->weightMeans = (float**)malloc(sizeof(float*) * layers);
    net->weightVars = (float**)malloc(sizeof(float*) * layers);
    net->weightSamples = (float**)malloc(sizeof(float*) * layers);
    
    net->weightGrads = (float**)malloc(sizeof(float*) * layers);
    
    net->weightMeanGradMeans = (float**)malloc(sizeof(float*) * layers);
    net->weightVarGradMeans = (float**)malloc(sizeof(float*) * layers);
    
    net->weightMeanGradVars = (float**)malloc(sizeof(float*) * layers);
    net->weightVarGradVars = (float**)malloc(sizeof(float*) * layers);
    
    for (uint32_t i = 0; i < layers + 1; i++) {
        cudaMalloc(&net->outputs[i], sizeof(float) * net->batchSize * parameters[i]);
        cudaMalloc(&net->outputGrads[i], sizeof(float) * net->batchSize * parameters[i]);
    }
    
    for (uint32_t i = 0; i < layers; i++) {
        cudaMalloc(&net->noisyWeights[i], sizeof(float) * parameters[i] * parameters[i + 1]);
        cudaMalloc(&net->weightMeans[i], sizeof(float) * parameters[i] * parameters[i + 1]);
        cudaMalloc(&net->weightVars[i], sizeof(float) * parameters[i] * parameters[i + 1]);
        cudaMalloc(&net->weightSamples[i], sizeof(float) * parameters[i] * parameters[i + 1]);
        
        cudaMalloc(&net->weightGrads[i], sizeof(float) * parameters[i] * parameters[i + 1]);
        
        cudaMalloc(&net->weightMeanGradMeans[i], sizeof(float) * parameters[i] * parameters[i + 1]);
        cudaMalloc(&net->weightVarGradMeans[i], sizeof(float) * parameters[i] * parameters[i + 1]);
        
        cudaMalloc(&net->weightMeanGradVars[i], sizeof(float) * parameters[i] * parameters[i + 1]);
        cudaMalloc(&net->weightVarGradVars[i], sizeof(float) * parameters[i] * parameters[i + 1]);
        
        float std = sqrtf(3.0f / parameters[i]);
        fillUniform(net->weightMeans[i], parameters[i] * parameters[i + 1], noise, -std, std);
        fill(net->weightVars[i], parameters[i] * parameters[i + 1], 0.017f);
        // fillUniform(net->weightVars[i], parameters[i] * parameters[i + 1], noise, 0.0f, 1.0f / parameters[i]);
        fill(net->weightMeanGradMeans[i], parameters[i] * parameters[i + 1], 0.0f);
        fill(net->weightVarGradMeans[i], parameters[i] * parameters[i + 1], 0.0f);
        fill(net->weightMeanGradVars[i], parameters[i] * parameters[i + 1], 0.0f);
        fill(net->weightVarGradVars[i], parameters[i] * parameters[i + 1], 0.0f);
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

void forwardNoiseless(cublasHandle_t *cublasHandle, Network* net) {
    const float ONE = 1.0f;
    const float ZERO = 0.0f;
    
    for (uint32_t i = 0; i < net->layers - 1; i++) {
        cublasSgemm(*cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
            net->parameters[i + 1], net->batchSize, net->parameters[i],
            &ONE,
            net->weightMeans[i], net->parameters[i + 1],
            net->outputs[i], net->parameters[i],
            &ZERO,
            net->outputs[i + 1], net->parameters[i + 1]);
            
        reluForward(net->outputs[i + 1], net->batchSize * net->parameters[i + 1]);
    }
    
    cublasSgemm(*cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
        net->parameters[net->layers], net->batchSize, net->parameters[net->layers - 1],
        &ONE,
        net->weightMeans[net->layers - 1], net->parameters[net->layers],
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
    // dTensor[idx] += net.learningRate * (grad - net.weightDecay * dTensor[idx]);
}

void integratedAdamUpdate(float *dTensor, float *dTensorGrad, float *dTensorMean, float *dTensorVar, uint32_t size, Network *net) {
    _integratedAdamUpdate<<<(size >> 10) + (size & 0x3ff ? 1 : 0), 0x400>>>(dTensor, dTensorGrad, dTensorMean, dTensorVar, size, *net);
}

void backwardNoiseless(cublasHandle_t *cublasHandle, Network* net) {
    const float ONE = 1.0f;
    const float ZERO = 0.0f;
    
    net->meanCor *= net->meanBeta;
    net->varCor *= net->varBeta;
    
    cublasSgemm(*cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
        net->parameters[net->layers], net->parameters[net->layers - 1], net->batchSize,
        &ONE,
        net->outputGrads[net->layers], net->parameters[net->layers],
        net->outputs[net->layers - 1], net->parameters[net->layers - 1],
        &ZERO,
        net->weightGrads[net->layers - 1], net->parameters[net->layers]);
        
    cublasSgemm(*cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
        net->parameters[net->layers - 1], net->batchSize, net->parameters[net->layers],
        &ONE,
        net->weightMeans[net->layers - 1], net->parameters[net->layers],
        net->outputGrads[net->layers], net->parameters[net->layers],
        &ZERO,
        net->outputGrads[net->layers - 1], net->parameters[net->layers - 1]);
                
    for (uint32_t i = net->layers - 1; i--;) {
        reluBackward(net->outputs[i + 1], net->outputGrads[i + 1], net->batchSize * net->parameters[i + 1]);
        cublasSgemm(*cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
            net->parameters[i + 1], net->parameters[i], net->batchSize,
            &ONE,
            net->outputGrads[i + 1], net->parameters[i + 1],
            net->outputs[i], net->parameters[i],
            &ZERO,
            net->weightGrads[i], net->parameters[i + 1]);
            
        cublasSgemm(*cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
            net->parameters[i], net->batchSize, net->parameters[i + 1],
            &ONE,
            net->weightMeans[i], net->parameters[i + 1],
            net->outputGrads[i + 1], net->parameters[i + 1],
            &ZERO,
            net->outputGrads[i], net->parameters[i]);
    }
    
    for (uint32_t i = 0; i < net->layers; i++) {
        integratedAdamUpdate(net->weightMeans[i], net->weightGrads[i], net->weightMeanGradMeans[i], net->weightMeanGradVars[i], net->parameters[i] * net->parameters[i + 1], net);
    }
}

__global__ void _fillGaussian(float* arr, uint32_t size, Noise noise, float mean, float variance) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        uint32_t hash = (index ^ noise.seed1) * 0x4ba1bb47;
        hash ^= (hash >> 17);
        float u1 = (float)hash / 0xffffffff;
        hash ^= (hash ^ noise.seed2) * 0xb7ebcb79;
        hash ^= (hash << 13);
        arr[index] = sqrtf(-2.0f * logf(u1)) * cosf(6.283185307179586476925286766559f / 0xffffffff * hash) * variance + mean;
    }
}

void fillGaussian(float* arr, uint32_t size, Noise* noise, float mean, float variance) {
    mix(noise);
    _fillGaussian<<<(size >> 10) + (size & 0x3ff ? 1 : 0), 1024>>>(arr, size, *noise, mean, variance);
}

__global__ void _genWeightNoise(float* weight, float* weightMean, float* weightVar, float* weightSample, uint32_t size) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        weight[index] = weightMean[index] + weightVar[index] * weightSample[index];
    }
}

void genWeightNoise(float* weight, float* weightMean, float* weightVar, float* weightSample, uint32_t size) {
    _genWeightNoise<<<(size >> 10) + (size & 0x3ff ? 1 : 0), 1024>>>(weight, weightMean, weightVar, weightSample, size);
}

void forwardNoisy(cublasHandle_t *cublasHandle, Network* net, Noise* noise, float noiseScale = 1.0f) {
    const float ONE = 1.0f;
    const float ZERO = 0.0f;
    
    for (uint32_t i = 0; i < net->layers - 1; i++) {
        // fillGaussian(net->weightSamples[i], net->parameters[i] * net->parameters[i + 1], noise, 0.0f, 1.0f);
        // genWeightNoise(net->noisyWeights[i], net->weightMeans[i], net->weightVars[i], net->weightSamples[i], net->parameters[i] * net->parameters[i + 1]);
        // cublasSgemm(*cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
        //     net->parameters[i + 1], net->batchSize, net->parameters[i],
        //     &ONE,
        //     net->noisyWeights[i], net->parameters[i + 1],
        //     net->outputs[i], net->parameters[i],
        //     &ZERO,
        //     net->outputs[i + 1], net->parameters[i + 1]);
        cublasSgemm(*cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
            net->parameters[i + 1], net->batchSize, net->parameters[i],
            &ONE,
            net->weightMeans[i], net->parameters[i + 1],
            net->outputs[i], net->parameters[i],
            &ZERO,
            net->outputs[i + 1], net->parameters[i + 1]);
            
        reluForward(net->outputs[i + 1], net->batchSize * net->parameters[i + 1]);
    }
    
    // fillGaussian(net->weightSamples[net->layers - 1], net->parameters[net->layers - 1] * net->parameters[net->layers], noise, 0.0f, 1.0f);
    // genWeightNoise(net->noisyWeights[net->layers - 1], net->weightMeans[net->layers - 1], net->weightVars[net->layers - 1], net->weightSamples[net->layers - 1], net->parameters[net->layers - 1] * net->parameters[net->layers]);
    // cublasSgemm(*cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
    //     net->parameters[net->layers], net->batchSize, net->parameters[net->layers - 1],
    //     &ONE,
    //     net->noisyWeights[net->layers - 1], net->parameters[net->layers],
    //     net->outputs[net->layers - 1], net->parameters[net->layers - 1],
    //     &ZERO,
    //     net->outputs[net->layers], net->parameters[net->layers]);
    
    fillGaussian(net->outputs[net->layers], net->parameters[net->layers] * net->batchSize, noise, 0.0f, noiseScale);
    cublasSgemm(*cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
        net->parameters[net->layers], net->batchSize, net->parameters[net->layers - 1],
        &ONE,
        net->weightMeans[net->layers - 1], net->parameters[net->layers],
        net->outputs[net->layers - 1], net->parameters[net->layers - 1],
        &ONE,
        net->outputs[net->layers], net->parameters[net->layers]);
}

__global__ void _integratedNoiseAdamUpdate(float *dTensor, float *dTensorGrad, float *dTensorSamples, float *dTensorMean, float *dTensorVar, uint32_t size, Network net) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float grad = dTensorGrad[idx] * dTensorSamples[idx];
    float mean = net.meanBeta * dTensorMean[idx] + (1.0f - net.meanBeta) * grad;
    float var = net.varBeta * dTensorVar[idx] + (1.0f - net.varBeta) * grad * grad;
    float meanCor = mean / (1.0f - net.meanCor);
    float varCor = var / (1.0f - net.varCor);
    dTensorMean[idx] = mean;
    dTensorVar[idx] = var;
    dTensor[idx] += net.learningRate * (meanCor / (sqrtf(varCor) + 1e-8f) - net.weightDecay * dTensor[idx]);
}

void integratedNoiseAdamUpdate(float *dTensor, float *dTensorGrad, float *dTensorSamples, float *dTensorMean, float *dTensorVar, uint32_t size, Network *net) {
    _integratedNoiseAdamUpdate<<<(size >> 10) + (size & 0x3ff ? 1 : 0), 0x400>>>(dTensor, dTensorGrad, dTensorSamples, dTensorMean, dTensorVar, size, *net);
}

void backwardNoisy(cublasHandle_t *cublasHandle, Network* net) {
    const float ONE = 1.0f;
    const float ZERO = 0.0f;
    
    net->meanCor *= net->meanBeta;
    net->varCor *= net->varBeta;
    
    cublasSgemm(*cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
        net->parameters[net->layers], net->parameters[net->layers - 1], net->batchSize,
        &ONE,
        net->outputGrads[net->layers], net->parameters[net->layers],
        net->outputs[net->layers - 1], net->parameters[net->layers - 1],
        &ZERO,
        net->weightGrads[net->layers - 1], net->parameters[net->layers]);
        
    cublasSgemm(*cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
        net->parameters[net->layers - 1], net->batchSize, net->parameters[net->layers],
        &ONE,
        net->noisyWeights[net->layers - 1], net->parameters[net->layers],
        net->outputGrads[net->layers], net->parameters[net->layers],
        &ZERO,
        net->outputGrads[net->layers - 1], net->parameters[net->layers - 1]);
                
    for (uint32_t i = net->layers - 1; i--;) {
        reluBackward(net->outputs[i + 1], net->outputGrads[i + 1], net->batchSize * net->parameters[i + 1]);
        cublasSgemm(*cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
            net->parameters[i + 1], net->parameters[i], net->batchSize,
            &ONE,
            net->outputGrads[i + 1], net->parameters[i + 1],
            net->outputs[i], net->parameters[i],
            &ZERO,
            net->weightGrads[i], net->parameters[i + 1]);
            
        cublasSgemm(*cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
            net->parameters[i], net->batchSize, net->parameters[i + 1],
            &ONE,
            net->noisyWeights[i], net->parameters[i + 1],
            net->outputGrads[i + 1], net->parameters[i + 1],
            &ZERO,
            net->outputGrads[i], net->parameters[i]);
    }
    
    for (uint32_t i = 0; i < net->layers; i++) {
        integratedAdamUpdate(net->weightMeans[i], net->weightGrads[i], net->weightMeanGradMeans[i], net->weightMeanGradVars[i], net->parameters[i] * net->parameters[i + 1], net);
        integratedNoiseAdamUpdate(net->weightVars[i], net->weightGrads[i], net->weightSamples[i], net->weightVarGradMeans[i], net->weightVarGradVars[i], net->parameters[i] * net->parameters[i + 1], net);
    }
}

void copyParams(Network* net, Network* net2) {
    for (uint32_t i = 0; i < net->layers; i++) {
        cudaMemcpy(net->weightMeans[i], net2->weightMeans[i], net->parameters[i] * net->parameters[i + 1] * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(net->weightVars[i], net2->weightVars[i], net->parameters[i] * net->parameters[i + 1] * sizeof(float), cudaMemcpyDeviceToDevice);
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
        printTensor(net->weightMeans[i], net->parameters[i + 1], net->parameters[i]);
    }
    printf("Output\n");
    printTensor(net->outputs[net->layers], net->parameters[net->layers], net->batchSize);
}

void printBackParams(Network* net) {
    for (uint32_t i = net->layers; i--;) {
        printf("Layer %d\n", i);
        printf("outputGrads\n");
        printTensor(net->outputGrads[i], net->parameters[i + 1], net->batchSize);
        printf("weightGrads\n");
        printTensor(net->weightGrads[i], net->parameters[i + 1], net->parameters[i]);
    }
    printf("outputGrads\n");
    printTensor(net->outputGrads[0], net->parameters[1], net->batchSize);
}