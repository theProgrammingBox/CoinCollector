#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/time.h>

#include <cuda_runtime.h>

#define LEARNING_RATE 0.01
#define BOARD_WIDTH 2
#define EPOCHS 1024
#define QUEUE_LENGTH 64
#define MAX_BATCH_SIZE 16
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

__global__ void _fillUniform(float* arr, uint32_t size, uint32_t seed1, uint32_t seed2) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        int32_t hash = index;
        hash ^= (hash ^ seed1) * 0x4ba1bb47;
        hash ^= (hash ^ seed2) * 0xb7ebcb79;
        hash ^= hash << 5;
        arr[index] = hash * 0.0000000004656612875245797f;
    }
}

void fillUniform(float* arr, uint32_t size, uint32_t* seed1, uint32_t* seed2) {
    mixSeed(seed1, seed2);
    _fillUniform<<<(size >> 10) + (size & 0x3ff), 1024>>>(arr, size, *seed1, *seed2);
}

__global__ void fillGaussian(float* arr, uint32_t size, uint32_t seed1, uint32_t seed2) {
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
        arr[index] = sqrtf(-2 * logf(u1)) * cosf(2 * 3.14159265358979323846f * u2);
    }
}

void fillGaussian(float* arr, uint32_t size, uint32_t* seed1, uint32_t* seed2) {
    mixSeed(seed1, seed2);
    fillGaussian<<<(size >> 10) + (size & 0x3ff), 1024>>>(arr, size, *seed1, *seed2);
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
    
    float* input;//[BOARD_SIZE * MAX_BATCH_SIZE];
    float* hidden;//[HIDDEN_LAYER_SIZE * MAX_BATCH_SIZE];
    float* output;//[ACTIONS * MAX_BATCH_SIZE];
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
    
    cudaMalloc((void**)&model->input, BOARD_SIZE * MAX_BATCH_SIZE * sizeof(float));
    cudaMalloc((void**)&model->hidden, HIDDEN_LAYER_SIZE * MAX_BATCH_SIZE * sizeof(float));
    cudaMalloc((void**)&model->output, ACTIONS * MAX_BATCH_SIZE * sizeof(float));
    
    fillUniform(model->weight1, BOARD_SIZE * HIDDEN_LAYER_SIZE, seed1, seed2);
    fillUniform(model->weight2, HIDDEN_LAYER_SIZE * ACTIONS, seed1, seed2);
    fillUniform(model->bias1, HIDDEN_LAYER_SIZE, seed1, seed2);
    fillUniform(model->bias2, ACTIONS, seed1, seed2);
    
    fillUniform(model->weight1Var, BOARD_SIZE * HIDDEN_LAYER_SIZE, seed1, seed2);
    fillUniform(model->weight2Var, HIDDEN_LAYER_SIZE * ACTIONS, seed1, seed2);
    fillUniform(model->bias1Var, HIDDEN_LAYER_SIZE, seed1, seed2);
    fillUniform(model->bias2Var, ACTIONS, seed1, seed2);
}

void newNoise(Model *model, uint32_t* seed1, uint32_t* seed2) {
    fillUniform(model->weight1Sample, BOARD_SIZE * HIDDEN_LAYER_SIZE, seed1, seed2);
    fillUniform(model->weight2Sample, HIDDEN_LAYER_SIZE * ACTIONS, seed1, seed2);
    fillUniform(model->bias1Sample, HIDDEN_LAYER_SIZE, seed1, seed2);
    fillUniform(model->bias2Sample, ACTIONS, seed1, seed2);
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
    printf("\n");
    free(arr);
}

void forward(uint8_t *boards, uint32_t batchSize, Model *model) {
    
}

int main(int argc, char *argv[])
{
    Model model;
    uint32_t seed1, seed2;
    initializeSeeds(&seed1, &seed2);
    // initializeModel(&model, &seed1, &seed2);
    // printTensor(model.weight1, BOARD_SIZE, HIDDEN_LAYER_SIZE);
    
    const uint32_t samples = 1 << 13;
    float* arr;
    cudaMalloc((void**)&arr, samples * sizeof(float));
    // fillUniform(arr, samples, &seed1, &seed2);
    fillGaussian(arr, samples, &seed1, &seed2);
    printHistogram(arr, samples);
    return 0;
    // uint32_t seed1, seed2;
    // initializeSeeds(&seed1, &seed2);
    
    uint8_t queueInitState[QUEUE_LENGTH * BOARD_SIZE]{};
    uint8_t queueResState[QUEUE_LENGTH * BOARD_SIZE]{};
    uint8_t queueAction[QUEUE_LENGTH]{};
    uint8_t queueReward[QUEUE_LENGTH]{};
    uint32_t queueIndex = 0;
    
    uint8_t batchInitState[MAX_BATCH_SIZE * BOARD_SIZE]{};
    uint8_t batchResState[MAX_BATCH_SIZE * BOARD_SIZE]{};
    uint8_t batchAction[MAX_BATCH_SIZE]{};
    uint8_t batchReward[MAX_BATCH_SIZE]{};
    
    uint8_t board[BOARD_SIZE];
    uint8_t x, y, cx, cy, action;
    
    uint32_t epoch;
    for (epoch = 0; epoch < EPOCHS; epoch++) {
        // reset, randomize, place player and coin on board, and store initial state
        memset(board, 0, BOARD_SIZE);
        x = mixSeed(&seed1, &seed2) % BOARD_WIDTH;
        y = mixSeed(&seed1, &seed2) % BOARD_WIDTH;
        do {
            cx = mixSeed(&seed1, &seed2) % BOARD_WIDTH;
            cy = mixSeed(&seed1, &seed2) % BOARD_WIDTH;
        } while (x == cx && y == cy);
        board[x + y * BOARD_WIDTH] = 1;
        board[cx + cy * BOARD_WIDTH] = 2;
        memcpy(queueInitState + queueIndex * BOARD_SIZE, board, BOARD_SIZE);
        
        // sample action using noisy dqn
        // action = forward(board);
        action = mixSeed(&seed1, &seed2) % ACTIONS;
        
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
        memcpy(queueResState + queueIndex * BOARD_SIZE, board, BOARD_SIZE);
        queueIndex *= ++queueIndex != QUEUE_LENGTH;
        
        // sample MAX_BATCH_SIZE random entries from queue if there are enough entries
        uint32_t batchSize;
        uint32_t queueUpperIndex;
        if (epoch + 1 < MAX_BATCH_SIZE) batchSize = epoch + 1;
        else batchSize = MAX_BATCH_SIZE;
        if (epoch + 1 < QUEUE_LENGTH) queueUpperIndex = epoch + 1;
        else queueUpperIndex = QUEUE_LENGTH;
        
        uint32_t queueSample;
        uint32_t tmp;
        for (queueSample = 0; queueSample < batchSize; queueSample++) {
            tmp = mixSeed(&seed1, &seed2) % queueUpperIndex;
            memcpy(batchInitState + queueSample * BOARD_SIZE, queueInitState + tmp * BOARD_SIZE, BOARD_SIZE);
            memcpy(batchResState + queueSample * BOARD_SIZE, queueResState + tmp * BOARD_SIZE, BOARD_SIZE);
            batchAction[queueSample] = queueAction[tmp];
            batchReward[queueSample] = queueReward[tmp];
        }
        
        // print each entry in batch
        printf("Epoch: %d\n", epoch);
        for (queueSample = 0; queueSample < batchSize; queueSample++) {
            for (x = 0; x < 2; x++) {
                for (y = 0; y < 2; y++) {
                    printf("%d ", batchInitState[queueSample * BOARD_SIZE + x + y * 2]);
                }
                printf("\n");
            }
            printf("\n");
            
            for (x = 0; x < 2; x++) {
                for (y = 0; y < 2; y++) {
                    printf("%d ", batchResState[queueSample * BOARD_SIZE + x + y * 2]);
                }
                printf("\n");
            }
            printf("A: %d R: %d\n\n\n", batchAction[queueSample], batchReward[queueSample]);
        }
    }
    
    return 0;
}