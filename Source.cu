#include "Network.cuh"

#define BOARD_WIDTH 4
#define BOARD_SIZE (BOARD_WIDTH * BOARD_WIDTH)
#define VIS_SIZE (BOARD_SIZE - 1)
#define ACTIONS 4
#define INPUTS (BOARD_SIZE + 1)
#define SCORE_SIZE 1000

#define QUEUE_SIZE 65536
#define MIN_QUEUE_SIZE 8192
#define BATCH_SIZE 512
#define LEARNING_RATE 0.001f
#define WEIGHT_DECAY 0.0000f
#define REWARD_DECAY 0.99f
#define EPOCHES 65536
#define UNFREEZE 4

int main(int argc, char **argv) {
    Noise noise;
    initNoise(&noise);
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    Network net, frozenNet;
    uint32_t parameters[] = {INPUTS, 16, 16, ACTIONS};
    uint32_t layers = sizeof(parameters) / sizeof(uint32_t) - 1;
    initNetwork(&net, parameters, layers, &noise, LEARNING_RATE, BATCH_SIZE > VIS_SIZE ? BATCH_SIZE : VIS_SIZE, WEIGHT_DECAY);
    initNetwork(&frozenNet, parameters, layers, &noise, LEARNING_RATE, BATCH_SIZE > VIS_SIZE ? BATCH_SIZE : VIS_SIZE, WEIGHT_DECAY);
    
    float* states = (float*)malloc(BOARD_SIZE * QUEUE_SIZE * sizeof(float));
    uint8_t* actions = (uint8_t*)malloc(QUEUE_SIZE * sizeof(uint8_t));
    float* rewards = (float*)malloc(QUEUE_SIZE * sizeof(float));
    float* nextStates = (float*)malloc(BOARD_SIZE * QUEUE_SIZE * sizeof(float));
    uint32_t queueIdx = 0;
    
    uint32_t* sampledIdxs = (uint32_t*)malloc(BATCH_SIZE * sizeof(uint32_t));
    float* outputs = (float*)malloc(ACTIONS * (BATCH_SIZE > VIS_SIZE ? BATCH_SIZE : VIS_SIZE) * sizeof(float));
    float* outputGrads = (float*)malloc(ACTIONS * BATCH_SIZE * sizeof(float));
    float* bestScores = (float*)malloc(BATCH_SIZE * sizeof(float));
    
    float board[BOARD_SIZE]{};
    float score[SCORE_SIZE]{};
    uint8_t px, py, cx, cy;
    uint32_t scoreIdx = 0;
    float scoreSum = 0.0f;
    
    px = genUint(&noise) % BOARD_WIDTH;
    py = genUint(&noise) % BOARD_WIDTH;
    do {
        cx = genUint(&noise) % BOARD_WIDTH;
        cy = genUint(&noise) % BOARD_WIDTH;
    } while (cx == px && cy == py);
    board[py * BOARD_WIDTH + px] = 1.0f;
    board[cy * BOARD_WIDTH + cx] = -1.0f;
    
    const float one = 1.0f;
    for (uint32_t epoch = 0; epoch < EPOCHES; epoch++) {
        memcpy(states + queueIdx * BOARD_SIZE, board, BOARD_SIZE * sizeof(float));
        
        net.batchSize = VIS_SIZE;
        board[py * BOARD_WIDTH + px] = 0.0f;
        uint32_t idx = 0;
        for (uint8_t pyy = 0; pyy < BOARD_WIDTH; pyy++) {
            for (uint8_t pxx = 0; pxx < BOARD_WIDTH; pxx++) {
                if (pxx == cx && pyy == cy) continue;
                board[pyy * BOARD_WIDTH + pxx] = 1.0f;
                cudaMemcpy(net.outputs[0] + idx * INPUTS, board, BOARD_SIZE * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(net.outputs[0] + idx * INPUTS + BOARD_SIZE, &one, sizeof(float), cudaMemcpyHostToDevice);
                idx++;
                board[pyy * BOARD_WIDTH + pxx] = 0.0f;
            }
        }
        board[py * BOARD_WIDTH + px] = 1.0f;
        // forwardNoisy(&handle, &net, &noise);
        forwardNoiseless(&handle, &net);
        cudaMemcpy(outputs, net.outputs[net.layers], ACTIONS * VIS_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
        float epsilon = (epoch / (EPOCHES * 0.5f));
        epsilon = epsilon > 1.0f ? 0.0f : (1 - epsilon) * 1;
        uint8_t action = 0;
        uint32_t pos = py * BOARD_WIDTH + px;
        uint8_t bias = pos > (cy * BOARD_WIDTH + cx);
        float bestScore = outputs[(pos - bias) * ACTIONS] + genNormal(&noise) * epsilon;
        // float bestScore = outputs[(pos - bias) * ACTIONS];
        for (uint8_t i = 1; i < ACTIONS; i++) {
            float sample = outputs[(pos - bias) * ACTIONS + i] + genNormal(&noise) * epsilon;
            // float sample = outputs[(pos - bias) * ACTIONS + i];
            if (sample > bestScore) {
                bestScore = sample;
                action = i;
            }
        }
        
        float maxScore = -INFINITY;
        float minScore = INFINITY;
        for (uint8_t i = 1; i < VIS_SIZE; i++) {
            float bestScore = outputs[i * ACTIONS];
            for (uint8_t j = 1; j < ACTIONS; j++) {
                if (outputs[i * ACTIONS + j] > bestScore) {
                    bestScore = outputs[i * ACTIONS + j];
                }
            }
            if (bestScore > maxScore) {
                maxScore = bestScore;
            }
            if (bestScore < minScore) {
                minScore = bestScore;
            }
        }
        printf("\033[H");
        printf("%d/%d\n", epoch, EPOCHES);
        idx = 0;
        for (uint8_t y = 0; y < BOARD_WIDTH; y++) {
            for (uint8_t x = 0; x < BOARD_WIDTH; x++) {
                if (x == cx && y == cy) {
                    printf("\x1b[38;2;255;255;0m");
                    printf("$$");
                } else {
                    uint8_t act = 0;
                    float bestScore = outputs[idx * ACTIONS];
                    for (uint8_t i = 1; i < ACTIONS; i++) {
                        if (outputs[idx * ACTIONS + i] > bestScore) {
                            bestScore = outputs[idx * ACTIONS + i];
                            act = i;
                        }
                    }
                    if (x == px && y == py) {
                        printf("\x1b[38;2;255;0;255m");
                    } else {
                        uint8_t g = (bestScore - minScore) / (maxScore - minScore) * 255;
                        printf("\x1b[38;2;%d;%d;0m", 255 - g, g);
                    }
                    switch (act) {
                        case 0: printf("<<"); break;
                        case 1: printf(">>"); break;
                        case 2: printf("^^"); break;
                        case 3: printf("vv"); break;
                    }
                    idx++;
                }
            }
            printf("\n");
        }
        printf("\x1b[38;2;255;255;255m");
        
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
            cx = genUint(&noise) % BOARD_WIDTH;
            cy = genUint(&noise) % BOARD_WIDTH;
        }
        board[cy * BOARD_WIDTH + cx] = -1.0f;
        memcpy(nextStates + queueIdx * BOARD_SIZE, board, BOARD_SIZE * sizeof(float));
        queueIdx *= ++queueIdx != QUEUE_SIZE;
        
        score[scoreIdx] = reward;
        scoreSum += reward;
        scoreIdx *= ++scoreIdx != SCORE_SIZE;
        scoreSum -= score[scoreIdx];
        uint32_t scoreIdxCap = epoch >= SCORE_SIZE ? SCORE_SIZE : epoch + 1;
        printf("\033[2KAverage score: %f\n", scoreSum / scoreIdxCap);
        printf("\033[2KMax score: %f\n", maxScore);
        printf("\033[2KMin score: %f\n", minScore);
        if (epoch > EPOCHES * 0.9 && scoreSum / scoreIdxCap > 0.31f) {
            struct timeval tv;
            tv.tv_sec = 0;
            tv.tv_usec = 1000000;
            select(0, NULL, NULL, NULL, &tv);
        }
        // printTensor(net.weightSamples[0], net.parameters[1], net.parameters[0]);
        // printTensor(net.weightVars[0], net.parameters[1], net.parameters[0]);
        // printTensor(net.weightMeans[0], net.parameters[1], net.parameters[0]);
        
        
        if (epoch + 1 < MIN_QUEUE_SIZE) continue;
        uint32_t idxCap = epoch + 1 >= QUEUE_SIZE ? QUEUE_SIZE : epoch + 1;
        for (uint32_t i = 0; i < BATCH_SIZE; i++) {
            sampledIdxs[i] = genUint(&noise) % idxCap;
        }
        
        if (epoch % UNFREEZE == 0) {
            copyParams(&frozenNet, &net);
        }
        
        for (uint32_t i = 0; i < BATCH_SIZE; i++) {
            cudaMemcpy(frozenNet.outputs[0] + i * INPUTS, nextStates + sampledIdxs[i] * BOARD_SIZE, BOARD_SIZE * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(frozenNet.outputs[0] + i * INPUTS + BOARD_SIZE, &one, sizeof(float), cudaMemcpyHostToDevice);
        }
        forwardNoiseless(&handle, &frozenNet);
        // forwardNoisy(&handle, &frozenNet, &noise);
        cudaMemcpy(outputs, frozenNet.outputs[frozenNet.layers], ACTIONS * BATCH_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
        
        for (uint32_t i = 0; i < BATCH_SIZE; i++) {
            // float bestScore = outputs[i * ACTIONS];
            // for (uint8_t j = 1; j < ACTIONS; j++) {
            //     if (outputs[i * ACTIONS + j] > bestScore) {
            //         bestScore = outputs[i * ACTIONS + j];
            //     }
            // }
            // bestScores[i] = bestScore;
            float avgScore = 0.0f;
            for (uint8_t j = 0; j < ACTIONS; j++) {
                avgScore += outputs[i * ACTIONS + j];
            }
            bestScores[i] = avgScore / ACTIONS;
        }
        
        net.batchSize = BATCH_SIZE;
        for (uint32_t i = 0; i < BATCH_SIZE; i++) {
            cudaMemcpy(net.outputs[0] + i * INPUTS, states + sampledIdxs[i] * BOARD_SIZE, BOARD_SIZE * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(net.outputs[0] + i * INPUTS + BOARD_SIZE, &one, sizeof(float), cudaMemcpyHostToDevice);
        }
        // forwardNoisy(&handle, &net, &noise);
        forwardNoiseless(&handle, &net);
        cudaMemcpy(outputs, net.outputs[net.layers], ACTIONS * BATCH_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
        memset(outputGrads, 0, ACTIONS * BATCH_SIZE * sizeof(float));
        float minGrad = INFINITY;
        float maxGrad = -INFINITY;
        for (uint32_t i = 0; i < BATCH_SIZE; i++) {
            outputGrads[i * ACTIONS + actions[sampledIdxs[i]]] = (rewards[sampledIdxs[i]] + REWARD_DECAY * bestScores[i]) - outputs[i * ACTIONS + actions[sampledIdxs[i]]];
            if (outputGrads[i * ACTIONS + actions[sampledIdxs[i]]] < minGrad) {
                minGrad = outputGrads[i * ACTIONS + actions[sampledIdxs[i]]];
            }
            if (outputGrads[i * ACTIONS + actions[sampledIdxs[i]]] > maxGrad) {
                maxGrad = outputGrads[i * ACTIONS + actions[sampledIdxs[i]]];
            }
        }
        printf("\033[2KMin grad: %f\n", minGrad);
        printf("\033[2KMax grad: %f\n", maxGrad);
        cudaMemcpy(net.outputGrads[net.layers], outputGrads, ACTIONS * BATCH_SIZE * sizeof(float), cudaMemcpyHostToDevice);
        // backwardNoisy(&handle, &net);
        backwardNoiseless(&handle, &net);
    }

    return 0;
}