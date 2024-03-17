#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/time.h>

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

int main(int argc, char *argv[])
{
    uint32_t seed1, seed2;
    initializeSeeds(&seed1, &seed2);
    
    const uint32_t boardWidth = 2;
    const uint32_t boardSize = boardWidth * boardWidth;
    const uint32_t epochs = 8;
    const uint32_t queueLength = 64;
    const uint32_t queueSamples = 16;
    uint8_t queueInitState[queueLength * boardSize]{};
    uint8_t queueResState[queueLength * boardSize]{};
    uint8_t queueAction[queueLength]{};
    uint8_t queueReward[queueLength]{};
    uint32_t queueIndex = 0;
    
    uint8_t batchInitState[queueSamples * boardSize]{};
    uint8_t batchResState[queueSamples * boardSize]{};
    uint8_t batchAction[queueSamples]{};
    uint8_t batchReward[queueSamples]{};
    
    uint8_t board[boardSize];
    uint8_t x, y, cx, cy, action;
    
    uint32_t epoch;
    for (epoch = 0; epoch < epochs; epoch++) {
        // reset, randomize, place player and coin on board, and store initial state
        memset(board, 0, boardSize);
        x = mixSeed(&seed1, &seed2) % boardWidth;
        y = mixSeed(&seed1, &seed2) % boardWidth;
        do {
            cx = mixSeed(&seed1, &seed2) % boardWidth;
            cy = mixSeed(&seed1, &seed2) % boardWidth;
        } while (x == cx && y == cy);
        board[x + y * boardWidth] = 1;
        board[cx + cy * boardWidth] = 2;
        memcpy(queueInitState + queueIndex * boardSize, board, boardSize);
        
        // sample random action or from model based on epsilon greedy
        action = forward(board);
        
        // apply action
        board[x + y * boardWidth] = 0;
        switch (action) {
            case 0: if (x > 0) x--; break;
            case 1: if (x < boardWidth - 1) x++; break;
            case 2: if (y > 0) y--; break;
            case 3: if (y < boardWidth - 1) y++; break;
        }
        board[x + y * boardWidth] = 1;
        
        // store action and reward
        queueAction[queueIndex] = action;
        queueReward[queueIndex] = x == cx && y == cy;
        
        // update state and store
        while (x == cx && y == cy) {
            cx = mixSeed(&seed1, &seed2) % boardWidth;
            cy = mixSeed(&seed1, &seed2) % boardWidth;
        }
        board[cx + cy * boardWidth] = 2;
        memcpy(queueResState + queueIndex * boardSize, board, boardSize);
        queueIndex *= ++queueIndex != queueLength;
        
        // sample queueSamples random entries from queue if there are enough entries
        uint32_t batchSize;
        uint32_t queueUpperIndex;
        if (epoch + 1 < queueSamples) batchSize = epoch + 1;
        else batchSize = queueSamples;
        if (epoch + 1 < queueLength) queueUpperIndex = epoch + 1;
        else queueUpperIndex = queueLength;
        
        uint32_t queueSample;
        uint32_t tmp;
        for (queueSample = 0; queueSample < batchSize; queueSample++) {
            tmp = mixSeed(&seed1, &seed2) % queueUpperIndex;
            memcpy(batchInitState + queueSample * boardSize, queueInitState + tmp * boardSize, boardSize);
            memcpy(batchResState + queueSample * boardSize, queueResState + tmp * boardSize, boardSize);
            batchAction[queueSample] = queueAction[tmp];
            batchReward[queueSample] = queueReward[tmp];
        }
        
        // // print each entry in batch
        // printf("Epoch: %d\n", epoch);
        // for (queueSample = 0; queueSample < batchSize; queueSample++) {
        //     for (x = 0; x < 2; x++) {
        //         for (y = 0; y < 2; y++) {
        //             printf("%d ", batchInitState[queueSample * boardSize + x + y * 2]);
        //         }
        //         printf("\n");
        //     }
        //     printf("\n");
            
        //     for (x = 0; x < 2; x++) {
        //         for (y = 0; y < 2; y++) {
        //             printf("%d ", batchResState[queueSample * boardSize + x + y * 2]);
        //         }
        //         printf("\n");
        //     }
        //     printf("A: %d R: %d\n\n\n", batchAction[queueSample], batchReward[queueSample]);
        // }
    }
    
    return 0;
}