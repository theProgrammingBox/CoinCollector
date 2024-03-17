#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/time.h>

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

struct Model {
    // mean, variance, and sample noise for weights and biases
    float weight1[BOARD_SIZE * HIDDEN_LAYER_SIZE];
    float weight2[HIDDEN_LAYER_SIZE * ACTIONS];
    float bias1[HIDDEN_LAYER_SIZE];
    float bias2[ACTIONS];
    
    float weight1Var[BOARD_SIZE * HIDDEN_LAYER_SIZE];
    float weight2Var[HIDDEN_LAYER_SIZE * ACTIONS];
    float bias1Var[HIDDEN_LAYER_SIZE];
    float bias2Var[ACTIONS];
    
    float weight1Sample[BOARD_SIZE * HIDDEN_LAYER_SIZE];
    float weight2Sample[HIDDEN_LAYER_SIZE * ACTIONS];
    float bias1Sample[HIDDEN_LAYER_SIZE];
    float bias2Sample[ACTIONS];
    
    float input[BOARD_SIZE * MAX_BATCH_SIZE];
    float hidden[HIDDEN_LAYER_SIZE * MAX_BATCH_SIZE];
    float output[ACTIONS * MAX_BATCH_SIZE];
};

void forward(uint8_t *boards, uint32_t batchSize, Model *model) {
    
}

__global__ void fill(float* arr, float val, uint32_t size) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) arr[index] = index;
}

int main(int argc, char *argv[])
{
    float* weight;
    cudaMalloc(&weight, sizeof(float) * 100);
    fill<<<1, 100>>>(weight, 0.0, 100);
    float weightHost[100];
    cudaMemcpy(weightHost, weight, sizeof(float) * 100, cudaMemcpyDeviceToHost);
    for (uint32_t i = 0; i < 100; i++) printf("%f\n", weightHost[i]);
    cudaFree(weight);
    
    return 0;
    uint32_t seed1, seed2;
    initializeSeeds(&seed1, &seed2);
    
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