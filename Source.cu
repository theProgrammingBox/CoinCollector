#include "Network.cuh"

#define BOARD_WIDTH 7
#define BOARD_SIZE (BOARD_WIDTH * BOARD_WIDTH)
#define ACTIONS 4
#define INPUTS (BOARD_SIZE + 1)
#define SCORE_SIZE 1000

#define QUEUE_SIZE 10000
#define MIN_QUEUE_SIZE 1000
#define BATCH_SIZE 64
#define LEARNING_RATE 0.001f
#define WEIGHT_DECAY 0.000f
#define REWARD_DECAY 0.99f
#define EPOCHES 100000

int main(int argc, char **argv) {
    Noise noise;
    initNoise(&noise);
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    Network net;
    uint32_t parameters[] = {INPUTS, 16, 16, ACTIONS};
    uint32_t layers = sizeof(parameters) / sizeof(uint32_t) - 1;
    initNetwork(&net, parameters, layers, &noise, LEARNING_RATE, BATCH_SIZE, WEIGHT_DECAY);
    
    float states[BOARD_SIZE * QUEUE_SIZE];
    uint8_t actions[QUEUE_SIZE];
    float rewards[QUEUE_SIZE];
    float nextStates[BOARD_SIZE * QUEUE_SIZE];
    uint32_t queueIdx = 0;
    
    uint32_t sampledIdxs[BATCH_SIZE];
    float outputs[ACTIONS * BATCH_SIZE];
    float outputGrads[ACTIONS * BATCH_SIZE];
    float bestScores[BATCH_SIZE];
    
    float board[BOARD_SIZE]{};
    float score[SCORE_SIZE]{};
    uint8_t px, py, cx, cy;
    uint32_t scoreIdx = 0;
    float scoreSum = 0.0f;
    
    px = genNoise(&noise) % BOARD_WIDTH;
    py = genNoise(&noise) % BOARD_WIDTH;
    do {
        cx = genNoise(&noise) % BOARD_WIDTH;
        cy = genNoise(&noise) % BOARD_WIDTH;
    } while (cx == px && cy == py);
    board[py * BOARD_WIDTH + px] = 1.0f;
    board[cy * BOARD_WIDTH + cx] = -1.0f;
    
    const float one = 1.0f;
    for (uint32_t epoch = 0; epoch < EPOCHES; epoch++) {
        memcpy(states + queueIdx * BOARD_SIZE, board, BOARD_SIZE * sizeof(float));
        
        uint8_t action;
        // if (genNoise(&noise) % 100 < 10) {
        //     action = genNoise(&noise) % ACTIONS;
        // } else {
            net.batchSize = 1;
            cudaMemcpy(net.outputs[0], board, BOARD_SIZE * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(net.outputs[0] + BOARD_SIZE, &one, sizeof(float), cudaMemcpyHostToDevice);
            forwardNoisy(&handle, &net, &noise);
            cudaMemcpy(outputs, net.outputs[net.layers], ACTIONS * sizeof(float), cudaMemcpyDeviceToHost);
            action = 0;
            for (uint8_t i = 1; i < ACTIONS; i++) {
                if (outputs[i] > outputs[action]) {
                    action = i;
                }
            }
        // }
        
        board[py * BOARD_WIDTH + px] = 0.0f;
        switch (action) {
            case 0: if (px > 0) px--; break;
            case 1: if (px < BOARD_WIDTH - 1) px++; break;
            case 2: if (py > 0) py--; break;
            case 3: if (py < BOARD_WIDTH - 1) py++; break;
        }
        board[py * BOARD_WIDTH + px] = 1.0f;
        float reward = cx == px && cy == py;
        
        actions[queueIdx] = action;
        rewards[queueIdx] = reward;
        
        while (cx == px && cy == py) {
            cx = genNoise(&noise) % BOARD_WIDTH;
            cy = genNoise(&noise) % BOARD_WIDTH;
        }
        board[cy * BOARD_WIDTH + cx] = -1.0f;
        memcpy(nextStates + queueIdx * BOARD_SIZE, board, BOARD_SIZE * sizeof(float));
        queueIdx *= ++queueIdx != QUEUE_SIZE;
        
        printf("\033[H\033[J");
        printf("%d/%d\n", epoch, EPOCHES);
        for (uint8_t y = 0; y < BOARD_WIDTH; y++) {
            for (uint8_t x = 0; x < BOARD_WIDTH; x++) {
                switch ((int)board[y * BOARD_WIDTH + x]) {
                    case 1: printf("||"); break;
                    case -1: printf("$$"); break;
                    default: printf(".."); break;
                }
            }
            printf("\n");
        }
        score[scoreIdx] = reward;
        scoreSum += reward;
        scoreIdx *= ++scoreIdx != SCORE_SIZE;
        scoreSum -= score[scoreIdx];
        printf("Average score: %f\n", scoreSum / SCORE_SIZE);
        
        if (epoch + 1 < MIN_QUEUE_SIZE) continue;
        uint32_t idxCap = epoch >= QUEUE_SIZE ? QUEUE_SIZE : epoch;
        for (uint32_t i = 0; i < BATCH_SIZE; i++) {
            sampledIdxs[i] = genNoise(&noise) % idxCap;
        }
        
        net.batchSize = BATCH_SIZE;
        for (uint32_t i = 0; i < BATCH_SIZE; i++) {
            cudaMemcpy(net.outputs[0] + i * INPUTS, nextStates + sampledIdxs[i] * BOARD_SIZE, BOARD_SIZE * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(net.outputs[0] + i * INPUTS + BOARD_SIZE, &one, sizeof(float), cudaMemcpyHostToDevice);
        }
        forwardNoiseless(&handle, &net);
        cudaMemcpy(outputs, net.outputs[net.layers], ACTIONS * BATCH_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
        
        float bestScore;
        for (uint32_t i = 0; i < BATCH_SIZE; i++) {
            bestScore = outputs[i * ACTIONS];
            for (uint8_t j = 1; j < ACTIONS; j++) {
                if (outputs[i * ACTIONS + j] > bestScore) {
                    bestScore = outputs[i * ACTIONS + j];
                }
            }
            bestScores[i] = bestScore;
        }
        
        for (uint32_t i = 0; i < BATCH_SIZE; i++) {
            cudaMemcpy(net.outputs[0] + i * INPUTS, states + sampledIdxs[i] * BOARD_SIZE, BOARD_SIZE * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(net.outputs[0] + i * INPUTS + BOARD_SIZE, &one, sizeof(float), cudaMemcpyHostToDevice);
        }
        forwardNoisy(&handle, &net, &noise);
        cudaMemcpy(outputs, net.outputs[net.layers], ACTIONS * BATCH_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
        memset(outputGrads, 0, ACTIONS * BATCH_SIZE * sizeof(float));
        for (uint32_t i = 0; i < BATCH_SIZE; i++) {
            outputGrads[i * ACTIONS + actions[sampledIdxs[i]]] = rewards[sampledIdxs[i]] + REWARD_DECAY * bestScores[i] - outputs[i * ACTIONS + actions[sampledIdxs[i]]];
        }
        cudaMemcpy(net.outputGrads[net.layers], outputGrads, ACTIONS * BATCH_SIZE * sizeof(float), cudaMemcpyHostToDevice);
        backwardNoisy(&handle, &net);
    }

    return 0;
}