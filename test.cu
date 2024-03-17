#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/time.h>

// #include <cuda_runtime.h>
#include <cublas_v2.h>

#define LEARNING_RATE 0.01
#define BOARD_WIDTH 2
#define EPOCHS 4096
#define QUEUE_LENGTH 1024
#define MAX_BATCH_SIZE 64
#define HIDDEN_LAYER_SIZE 16
#define ACTIONS 4
#define BOARD_SIZE (BOARD_WIDTH * BOARD_WIDTH)

/*
TODO:
- add forward and backward function
- add adam optimizer
*/

uint32_t mixSeed(uint32_t *seed1, uint32_t *seed2) {
    *seed2 ^= (*seed1 >> 17) * 0xbf324c81;
    *seed1 ^= (*seed2 >> 13) * 0x9c7493ad;
    return *seed1;
}

void initializeSeeds(uint32_t *seed1, uint32_t *seed2) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    *seed1 = tv.tv_sec;
    *seed2 = tv.tv_usec;
    for (uint8_t i = 8; i--;) mixSeed(seed1, seed2);
}

__global__ void _fillUniform(float* arr, uint32_t size, uint32_t seed1, uint32_t seed2, float lowerBound, float upperBound) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        uint32_t hash = index;
        hash ^= (hash ^ seed1) * 0x4ba1bb47;
        hash ^= (hash ^ seed2) * 0xb7ebcb79;
        hash ^= hash << 5;
        arr[index] = hash * 0.00000000023283064365386962890625f * (upperBound - lowerBound) + lowerBound;
    }
}

void fillUniform(float* arr, uint32_t size, uint32_t* seed1, uint32_t* seed2, float lowerBound, float upperBound) {
    mixSeed(seed1, seed2);
    _fillUniform<<<(size >> 10) + (size & 0x3ff ? 1 : 0), 1024>>>(arr, size, *seed1, *seed2, lowerBound, upperBound);
}

__global__ void _fillGaussian(float* arr, uint32_t size, uint32_t seed1, uint32_t seed2, float mean, float variance) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    // Box-Muller transform
    if (index < size) {
        uint32_t hash = index;
        hash ^= (hash ^ seed1) * 0x4ba1bb47;
        hash ^= hash << 5;
        float u1 = hash * 0.00000000023283064365386962890625f;
        hash ^= (hash ^ seed2) * 0xb7ebcb79;
        hash ^= hash << 5;
        float u2 = hash * 0.00000000023283064365386962890625f;
        arr[index] = sqrtf(-2 * logf(u1)) * cosf(u2 * 6.283185307179586476925286766559f) * variance + mean;
    }
}

void fillGaussian(float* arr, uint32_t size, uint32_t* seed1, uint32_t* seed2, float mean, float variance) {
    mixSeed(seed1, seed2);
    _fillGaussian<<<(size >> 10) + (size & 0x3ff ? 1 : 0), 1024>>>(arr, size, *seed1, *seed2, mean, variance);
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

__global__ void newWeights(float* weights, float* weightsVar, float* weightsSample, float* newWeights, uint32_t size) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        newWeights[index] = weights[index] + weightsVar[index] * weightsSample[index];
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

__global__ void _reluBackward(float *dTensor, float *dTensorGrad, uint32_t size) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    dTensorGrad[idx] = dTensor[idx] > 0 ? dTensorGrad[idx] : 0;
}

void reluBackward(float *dTensor, float *dTensorGrad, uint32_t size) {
    _reluBackward<<<(size >> 10) + (size & 0x3ff), 0x400>>>(dTensor, dTensorGrad, size);
}

__global__ void _add(float *dTensor, float *tensor2, uint32_t size, float alpha) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    dTensor[idx] += tensor2[idx] * alpha;
}

void add(float *dTensor, float *tensor2, uint32_t size, float alpha) {
    _add<<<(size >> 10) + (size & 0x3ff), 0x400>>>(dTensor, tensor2, size, alpha);
}

__global__ void __mul(float *dTensor, float *tensor2, uint32_t size) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    dTensor[idx] *= tensor2[idx];
}

void mul(float *dTensor, float *tensor2, uint32_t size) {
    __mul<<<(size >> 10) + (size & 0x3ff), 0x400>>>(dTensor, tensor2, size);
}

struct Model {
    // mean, variance, and sample noise for weights and biases
    float* weight1;//[BOARD_SIZE * HIDDEN_LAYER_SIZE];
    float* weight2;//[HIDDEN_LAYER_SIZE * ACTIONS];
    float* bias1;//[HIDDEN_LAYER_SIZE];
    float* bias2;//[ACTIONS];
    
    float* weight1Var;//[BOARD_SIZE * HIDDEN_LAYER_SIZE];
    float* weight2Var;//[HIDDEN_LAYER_SIZE * ACTIONS];
    float* bias1Var;//[HIDDEN_LAYER_SIZE];
    float* bias2Var;//[ACTIONS];
    
    float* weight1Sample;//[BOARD_SIZE * HIDDEN_LAYER_SIZE];
    float* weight2Sample;//[HIDDEN_LAYER_SIZE * ACTIONS];
    float* bias1Sample;//[HIDDEN_LAYER_SIZE];
    float* bias2Sample;//[ACTIONS];
    
    float* newWeight1;//[BOARD_SIZE * HIDDEN_LAYER_SIZE];
    float* newWeight2;//[HIDDEN_LAYER_SIZE * ACTIONS];
    float* newBias1;//[HIDDEN_LAYER_SIZE];
    float* newBias2;//[ACTIONS];
    
    float* input;//[BOARD_SIZE * MAX_BATCH_SIZE];
    float* hidden;//[HIDDEN_LAYER_SIZE * MAX_BATCH_SIZE];
    float* output;//[ACTIONS * MAX_BATCH_SIZE];
    
    float* weight1Grad;//[BOARD_SIZE * HIDDEN_LAYER_SIZE];
    float* weight2Grad;//[HIDDEN_LAYER_SIZE * ACTIONS];
    float* bias1Grad;//[HIDDEN_LAYER_SIZE];
    float* bias2Grad;//[ACTIONS];
    
    float* weight1VarGrad;//[BOARD_SIZE * HIDDEN_LAYER_SIZE];
    float* weight2VarGrad;//[HIDDEN_LAYER_SIZE * ACTIONS];
    float* bias1VarGrad;//[HIDDEN_LAYER_SIZE];
    float* bias2VarGrad;//[ACTIONS];
    
    float* hiddenGrad;//[HIDDEN_LAYER_SIZE * MAX_BATCH_SIZE];
    float* outputGrad;//[BOARD_SIZE * MAX_BATCH_SIZE];
};

void initializeModel(Model *model, uint32_t* seed1, uint32_t* seed2) {
    cudaMalloc((void**)&model->weight1, BOARD_SIZE * HIDDEN_LAYER_SIZE * sizeof(float));
    cudaMalloc((void**)&model->weight2, HIDDEN_LAYER_SIZE * ACTIONS * sizeof(float));
    cudaMalloc((void**)&model->bias1, HIDDEN_LAYER_SIZE * sizeof(float));
    cudaMalloc((void**)&model->bias2, ACTIONS * sizeof(float));
    
    cudaMalloc((void**)&model->weight1Var, BOARD_SIZE * HIDDEN_LAYER_SIZE * sizeof(float));
    cudaMalloc((void**)&model->weight2Var, HIDDEN_LAYER_SIZE * ACTIONS * sizeof(float));
    cudaMalloc((void**)&model->bias1Var, HIDDEN_LAYER_SIZE * sizeof(float));
    cudaMalloc((void**)&model->bias2Var, ACTIONS * sizeof(float));
    
    cudaMalloc((void**)&model->weight1Sample, BOARD_SIZE * HIDDEN_LAYER_SIZE * sizeof(float));
    cudaMalloc((void**)&model->weight2Sample, HIDDEN_LAYER_SIZE * ACTIONS * sizeof(float));
    cudaMalloc((void**)&model->bias1Sample, HIDDEN_LAYER_SIZE * sizeof(float));
    cudaMalloc((void**)&model->bias2Sample, ACTIONS * sizeof(float));
    
    cudaMalloc((void**)&model->newWeight1, BOARD_SIZE * HIDDEN_LAYER_SIZE * sizeof(float));
    cudaMalloc((void**)&model->newWeight2, HIDDEN_LAYER_SIZE * ACTIONS * sizeof(float));
    cudaMalloc((void**)&model->newBias1, HIDDEN_LAYER_SIZE * sizeof(float));
    cudaMalloc((void**)&model->newBias2, ACTIONS * sizeof(float));
    
    cudaMalloc((void**)&model->input, BOARD_SIZE * MAX_BATCH_SIZE * sizeof(float));
    cudaMalloc((void**)&model->hidden, HIDDEN_LAYER_SIZE * MAX_BATCH_SIZE * sizeof(float));
    cudaMalloc((void**)&model->output, ACTIONS * MAX_BATCH_SIZE * sizeof(float));
    
    fillUniform(model->weight1, BOARD_SIZE * HIDDEN_LAYER_SIZE, seed1, seed2, -0.2, 0.2);
    fillUniform(model->weight2, HIDDEN_LAYER_SIZE * ACTIONS, seed1, seed2, -0.2, 0.2);
    fillUniform(model->bias1, HIDDEN_LAYER_SIZE, seed1, seed2, -0.1, 0.1);
    fillUniform(model->bias2, ACTIONS, seed1, seed2, -0.1, 0.1);
    
    // fillUniform(model->weight1Var, BOARD_SIZE * HIDDEN_LAYER_SIZE, seed1, seed2, 0, 1);
    // fillUniform(model->weight2Var, HIDDEN_LAYER_SIZE * ACTIONS, seed1, seed2, 0, 1);
    // fillUniform(model->bias1Var, HIDDEN_LAYER_SIZE, seed1, seed2, 0, 1);
    // fillUniform(model->bias2Var, ACTIONS, seed1, seed2, 0, 1);
    
    fill(model->weight1Var, BOARD_SIZE * HIDDEN_LAYER_SIZE, 0.4);
    fill(model->weight2Var, HIDDEN_LAYER_SIZE * ACTIONS, 0.4);
    fill(model->bias1Var, HIDDEN_LAYER_SIZE, 0.4);
    fill(model->bias2Var, ACTIONS, 0.4);
}

void newNoise(Model *model, uint32_t* seed1, uint32_t* seed2) {
    fillGaussian(model->weight1Sample, BOARD_SIZE * HIDDEN_LAYER_SIZE, seed1, seed2, 0, 1);
    fillGaussian(model->weight2Sample, HIDDEN_LAYER_SIZE * ACTIONS, seed1, seed2, 0, 1);
    fillGaussian(model->bias1Sample, HIDDEN_LAYER_SIZE, seed1, seed2, 0, 1);
    fillGaussian(model->bias2Sample, ACTIONS, seed1, seed2, 0, 1);
    
    newWeights<<<(BOARD_SIZE * HIDDEN_LAYER_SIZE >> 10) + (BOARD_SIZE * HIDDEN_LAYER_SIZE & 0x3ff? 1 : 0), 1024>>>(model->weight1, model->weight1Var, model->weight1Sample, model->newWeight1, BOARD_SIZE * HIDDEN_LAYER_SIZE);
    newWeights<<<(HIDDEN_LAYER_SIZE * ACTIONS >> 10) + (HIDDEN_LAYER_SIZE * ACTIONS & 0x3ff ? 1 : 0), 1024>>>(model->weight2, model->weight2Var, model->weight2Sample, model->newWeight2, HIDDEN_LAYER_SIZE * ACTIONS);
}

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

void printHistogram(float* tensor, uint32_t size) {
    float* arr = (float*)malloc(size * sizeof(float));
    cudaMemcpy(arr, tensor, size * sizeof(float), cudaMemcpyDeviceToHost);
    // from -4 to 4
    uint32_t hist[32]{};
    for (uint32_t i = 0; i < size; i++) {
        hist[(int)(arr[i] * 4 + 16)]++;
    }
    for (uint32_t i = 0; i < 32; i++) {
        printf("%f ", (float)hist[i] / size);
    }
    printf("\n\n");
    free(arr);
}

void forward(cublasHandle_t* handle, uint32_t batchSize, Model *model) {
    const float ONE = 1;
    const float ZERO = 0;
    
    cublasSgemm(
        *handle, CUBLAS_OP_N, CUBLAS_OP_N,
        HIDDEN_LAYER_SIZE, batchSize, BOARD_SIZE,
        &ONE,
        model->newWeight1, HIDDEN_LAYER_SIZE,
        model->input, BOARD_SIZE,
        &ZERO,
        model->hidden, HIDDEN_LAYER_SIZE
    );
    
    // cublasSgeam(
    //     *handle, CUBLAS_OP_N, CUBLAS_OP_N,
    //     HIDDEN_LAYER_SIZE, batchSize,
    //     &ONE,
    //     model->hidden, HIDDEN_LAYER_SIZE,
    //     &ONE,
    //     model->bias1, HIDDEN_LAYER_SIZE,
    //     model->hidden, HIDDEN_LAYER_SIZE
    // );
    
    add(model->hidden, model->bias1, HIDDEN_LAYER_SIZE * batchSize, ONE);
    
    reluForward(model->hidden, HIDDEN_LAYER_SIZE * batchSize);
    
    cublasSgemm(
        *handle, CUBLAS_OP_N, CUBLAS_OP_N,
        ACTIONS, batchSize, HIDDEN_LAYER_SIZE,
        &ONE,
        model->newWeight2, ACTIONS,
        model->hidden, HIDDEN_LAYER_SIZE,
        &ZERO,
        model->output, ACTIONS
    );
    
    // cublasSgeam(
    //     *handle, CUBLAS_OP_N, CUBLAS_OP_N,
    //     ACTIONS, batchSize,
    //     &ONE,
    //     model->output, ACTIONS,
    //     &ONE,
    //     model->bias2, ACTIONS,
    //     model->output, ACTIONS
    // );
    
    add(model->output, model->bias2, ACTIONS * batchSize, ONE);
}

void backward(cublasHandle_t* handle, uint32_t batchSize, Model *model) {
    const float ONE = 1;
    const float ZERO = 0;
    
    cudaMemcpy(model->bias2Grad, model->outputGrad, ACTIONS * batchSize * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(model->bias2VarGrad, model->bias2Grad, ACTIONS * batchSize * sizeof(float), cudaMemcpyDeviceToDevice);
    mul(model->bias2VarGrad, model->bias2Sample, ACTIONS * batchSize);
    
    cublasSgemm(
        *handle, CUBLAS_OP_N, CUBLAS_OP_T,
        ACTIONS, HIDDEN_LAYER_SIZE, batchSize,
        &ONE,
        model->outputGrad, ACTIONS,
        model->hidden, HIDDEN_LAYER_SIZE,
        &ZERO,
        model->weight2Grad, ACTIONS
    );
    
    add(model->weight2, model->weight2Var, HIDDEN_LAYER_SIZE * ACTIONS, LEARNING_RATE);
    
    cudaMemcpy(model->weight2VarGrad, model->weight2Grad, HIDDEN_LAYER_SIZE * ACTIONS * sizeof(float), cudaMemcpyDeviceToDevice);
    mul(model->weight2VarGrad, model->weight2Sample, HIDDEN_LAYER_SIZE * ACTIONS);
    add(model->weight2Var, model->weight2VarGrad, HIDDEN_LAYER_SIZE * ACTIONS, LEARNING_RATE);
    
    cublasSgemm(
        *handle, CUBLAS_OP_T, CUBLAS_OP_N,
        HIDDEN_LAYER_SIZE, batchSize, ACTIONS,
        &ONE,
        model->weight2, ACTIONS,
        model->outputGrad, ACTIONS,
        &ZERO,
        model->hiddenGrad, HIDDEN_LAYER_SIZE
    );
    
    reluBackward(model->hidden, model->hiddenGrad, HIDDEN_LAYER_SIZE * batchSize);
    
    cudaMemcpy(model->bias1Grad, model->hiddenGrad, HIDDEN_LAYER_SIZE * batchSize * sizeof(float), cudaMemcpyDeviceToDevice);
    add(model->bias1, model->bias1Grad, HIDDEN_LAYER_SIZE, LEARNING_RATE);
    cudaMemcpy(model->bias1VarGrad, model->bias1Grad, HIDDEN_LAYER_SIZE * batchSize * sizeof(float), cudaMemcpyDeviceToDevice);
    mul(model->bias1VarGrad, model->bias1Sample, HIDDEN_LAYER_SIZE * batchSize);
    add(model->bias1Var, model->bias1VarGrad, HIDDEN_LAYER_SIZE, LEARNING_RATE);
    
    cublasSgemm(
        *handle, CUBLAS_OP_N, CUBLAS_OP_T,
        HIDDEN_LAYER_SIZE, BOARD_SIZE, batchSize,
        &ONE,
        model->hiddenGrad, HIDDEN_LAYER_SIZE,
        model->input, BOARD_SIZE,
        &ZERO,
        model->weight1Grad, HIDDEN_LAYER_SIZE
    );
    
    add(model->weight1, model->weight1Var, BOARD_SIZE * HIDDEN_LAYER_SIZE, LEARNING_RATE);
    
    cudaMemcpy(model->weight1VarGrad, model->weight1Grad, BOARD_SIZE * HIDDEN_LAYER_SIZE * sizeof(float), cudaMemcpyDeviceToDevice);
    mul(model->weight1VarGrad, model->weight1Sample, BOARD_SIZE * HIDDEN_LAYER_SIZE);
    add(model->weight1Var, model->weight1VarGrad, BOARD_SIZE * HIDDEN_LAYER_SIZE, LEARNING_RATE);
}

int main(int argc, char *argv[])
{
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    // for random number generation
    uint32_t seed1, seed2;
    initializeSeeds(&seed1, &seed2);
    
    Model model;
    initializeModel(&model, &seed1, &seed2);
    
    uint32_t queueIndex = 0;
    float queueInitState[QUEUE_LENGTH * BOARD_SIZE]{};
    float queueResState[QUEUE_LENGTH * BOARD_SIZE]{};
    uint8_t queueAction[QUEUE_LENGTH]{};
    float queueReward[QUEUE_LENGTH]{};
    
    float actions[ACTIONS];
    uint32_t sampleIndex[MAX_BATCH_SIZE];
    
    // float batchInitState[MAX_BATCH_SIZE * BOARD_SIZE]{};
    // float batchResState[MAX_BATCH_SIZE * BOARD_SIZE]{};
    // float batchAction[MAX_BATCH_SIZE]{};
    // float batchReward[MAX_BATCH_SIZE]{};
    
    float board[BOARD_SIZE];
    uint8_t x, y, cx, cy;
    uint8_t action;
    
    uint32_t epoch;
    for (epoch = 0; epoch < EPOCHS; epoch++) {
        // reset, randomize, place player and coin on board, and store initial state
        memset(board, 0, BOARD_SIZE * sizeof(float));
        x = mixSeed(&seed1, &seed2) % BOARD_WIDTH;
        y = mixSeed(&seed1, &seed2) % BOARD_WIDTH;
        do {
            cx = mixSeed(&seed1, &seed2) % BOARD_WIDTH;
            cy = mixSeed(&seed1, &seed2) % BOARD_WIDTH;
        } while (x == cx && y == cy);
        board[x + y * BOARD_WIDTH] = 1;
        board[cx + cy * BOARD_WIDTH] = 2;
        memcpy(queueInitState + queueIndex * BOARD_SIZE, board, BOARD_SIZE * sizeof(float));
        
        // sample action using noisy dqn
        newNoise(&model, &seed1, &seed2);
        cudaMemcpy(model.input, board, BOARD_SIZE * sizeof(float), cudaMemcpyHostToDevice);
        forward(&handle, 1, &model);
        cudaMemcpy(actions, model.output, ACTIONS * sizeof(float), cudaMemcpyDeviceToHost);
        action = 0;
        for (uint8_t i = 1; i < ACTIONS; i++) {
            if (actions[i] > actions[action]) action = i;
        }
        // printTensor(model.output, 1, ACTIONS);
        // action = mixSeed(&seed1, &seed2) % ACTIONS;
        
        // apply action
        board[x + y * BOARD_WIDTH] = 0;
        switch (action) {
            case 0: if (x > 0) x--; break;
            case 1: if (x < BOARD_WIDTH - 1) x++; break;
            case 2: if (y > 0) y--; break;
            case 3: if (y < BOARD_WIDTH - 1) y++; break;
        }
        board[x + y * BOARD_WIDTH] = 1;
        
        // store action and reward
        queueAction[queueIndex] = action;
        queueReward[queueIndex] = x == cx && y == cy;
        
        // update state and store
        while (x == cx && y == cy) {
            cx = mixSeed(&seed1, &seed2) % BOARD_WIDTH;
            cy = mixSeed(&seed1, &seed2) % BOARD_WIDTH;
        }
        board[cx + cy * BOARD_WIDTH] = 2;
        memcpy(queueResState + queueIndex * BOARD_SIZE, board, BOARD_SIZE * sizeof(float));
        queueIndex *= ++queueIndex != QUEUE_LENGTH;
        
        // sample MAX_BATCH_SIZE random entries from queue if there are enough entries
        uint32_t batchSize;
        uint32_t queueUpperIndex;
        if (epoch + 1 < MAX_BATCH_SIZE) batchSize = epoch + 1;
        else batchSize = MAX_BATCH_SIZE;
        if (epoch + 1 < QUEUE_LENGTH) queueUpperIndex = epoch + 1;
        else queueUpperIndex = QUEUE_LENGTH;
        
        // adding random entries to batch
        uint32_t queueSample;
        uint32_t tmp;
        printf("Epoch: %d\n", epoch);
        for (queueSample = 0; queueSample < batchSize; queueSample++) {
            tmp = mixSeed(&seed1, &seed2) % queueUpperIndex;
            sampleIndex[queueSample] = tmp;
            cudaMemcpy(model.input + queueSample * BOARD_SIZE, queueInitState + tmp * BOARD_SIZE, BOARD_SIZE * sizeof(float), cudaMemcpyHostToDevice);
            // cudaMemcpy(model.input + queueSample * BOARD_SIZE, queueResState + tmp * BOARD_SIZE, BOARD_SIZE * sizeof(float), cudaMemcpyHostToDevice);
            // cudaMemcpy(model.input + queueSample, queueAction + tmp, sizeof(float), cudaMemcpyHostToDevice);
            // cudaMemcpy(model.input + queueSample, queueReward + tmp, sizeof(float), cudaMemcpyHostToDevice);
            
            // for (x = 0; x < 2; x++) {
            //     for (y = 0; y < 2; y++) {
            //         printf("%.0f ", queueInitState[tmp * BOARD_SIZE + x + y * 2]);
            //     }
            //     printf("\n");
            // }
            // printf("\n");
            
            // for (x = 0; x < 2; x++) {
            //     for (y = 0; y < 2; y++) {
            //         printf("%.0f ", queueResState[tmp * BOARD_SIZE + x + y * 2]);
            //     }
            //     printf("\n");
            // }
            
            // printf("A: %d R: %.0f\n\n\n", queueAction[tmp], queueReward[tmp]);
        }
        
        // forward propogate batch
        newNoise(&model, &seed1, &seed2);
        forward(&handle, batchSize, &model);
        backward(&handle, batchSize, &model);
    }
    
    return 0;
}