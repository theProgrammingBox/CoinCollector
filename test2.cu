#include "Network.cuh"
#include "Keyboard.cuh"

#define BOARD_WIDTH 4
#define BOARD_SIZE (BOARD_WIDTH * BOARD_WIDTH)
#define ACTIONS 4
#define NUM_FINAL_STATES (BOARD_SIZE * (BOARD_SIZE - 1) * ACTIONS)

#define DECAY 0.8

int main(int argc, char *argv[])
{
    Noise noise;
    initializeNoise(&noise);
    
    float states[BOARD_SIZE * NUM_FINAL_STATES];
    uint8_t actions[NUM_FINAL_STATES];
    float rewards[NUM_FINAL_STATES];
    float nextStates[BOARD_SIZE * NUM_FINAL_STATES];
    
    uint32_t queueIdx = 0;
    float board[BOARD_SIZE];
    for (uint8_t py = 0; py < BOARD_WIDTH; py++) {
        for (uint8_t px = 0; px < BOARD_WIDTH; px++) {
            for (uint8_t cy = 0; cy < BOARD_WIDTH; cy++) {
                for (uint8_t cx = 0; cx < BOARD_WIDTH; cx++) {
                    if (px == cx && py == cy) continue;
                    board[py * BOARD_WIDTH + px] = 1;
                    board[cy * BOARD_WIDTH + cx] = -1;
                    for (uint8_t a = 0; a < ACTIONS; a++) {
                        memcpy(states + queueIdx * ACTIONS * BOARD_SIZE + a * BOARD_SIZE, board, BOARD_SIZE * sizeof(float));
                        uint8_t pxx = px;
                        uint8_t pyy = py;
                        switch (a) {
                            case 0: if (pxx > 0) pxx--; break;
                            case 1: if (pxx < BOARD_WIDTH - 1) pxx++; break;
                            case 2: if (pyy > 0) pyy--; break;
                            case 3: if (pyy < BOARD_WIDTH - 1) pyy++; break;
                        }
                        board[py * BOARD_WIDTH + px] = 0;
                        board[pyy * BOARD_WIDTH + pxx] = 1;
                        uint8_t cxx = cx;
                        uint8_t cyy = cy;
                        while ((pxx == cxx) && (pyy == cyy)) {
                            cxx = genNoise(&noise) % BOARD_WIDTH;
                            cyy = genNoise(&noise) % BOARD_WIDTH;
                        }
                        board[cyy * BOARD_WIDTH + cxx] = -1;
                        actions[queueIdx * ACTIONS + a] = a;
                        rewards[queueIdx * ACTIONS + a] = (pxx == cx) && (pyy == cy);
                        memcpy(nextStates + queueIdx * ACTIONS * BOARD_SIZE + a * BOARD_SIZE, board, BOARD_SIZE * sizeof(float));
                        board[cyy * BOARD_WIDTH + cxx] = 0;
                        board[pyy * BOARD_WIDTH + pxx] = 0;
                        board[py * BOARD_WIDTH + px] = 1;
                        board[cy * BOARD_WIDTH + cx] = -1;
                    }
                    queueIdx++;
                    board[py * BOARD_WIDTH + px] = 0;
                    board[cy * BOARD_WIDTH + cx] = 0;
                }
            }
        }
    }
    
    struct Keyboard keyboard;
    initKeyboard(&keyboard);
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    Network net, net2;
    float learningRate = 0.001f;
    float weightDecay = 0.00f;
    uint32_t parameters[] = {BOARD_SIZE + 1, 16, 16, ACTIONS};
    uint32_t layers = sizeof(parameters) / sizeof(uint32_t) - 1;
    initNetwork(&net, parameters, layers, &noise, learningRate, NUM_FINAL_STATES, weightDecay);
    initNetwork(&net2, parameters, layers, &noise, learningRate, NUM_FINAL_STATES, weightDecay);
    
    const uint32_t EPOCHES = 1 << 14;
    float one = 1;
    // for (uint32_t epoch = 0; epoch < EPOCHES; epoch++) {
    uint8_t cx = 0;
    uint8_t cy = 0;
    memset(board, 0, BOARD_SIZE * sizeof(float));
    board[cy * BOARD_WIDTH + cx] = -1;
    for (uint32_t epoch = 0; true; epoch++) {
        if (epoch % 4 == 0) {
            copyParams(&net, &net2);
        }
        for (uint32_t i = 0; i < NUM_FINAL_STATES; i++) {
            cudaMemcpy(net2.outputs[0] + i * (BOARD_SIZE + 1), nextStates + i * BOARD_SIZE, BOARD_SIZE * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(net2.outputs[0] + i * (BOARD_SIZE + 1) + BOARD_SIZE, &one, sizeof(float), cudaMemcpyHostToDevice);
        }
        forwardPropagate(&handle, &net2);
        float output[NUM_FINAL_STATES * ACTIONS];
        cudaMemcpy(output, net2.outputs[net2.layers], NUM_FINAL_STATES * ACTIONS * sizeof(float), cudaMemcpyDeviceToHost);
        float nextBestScore[NUM_FINAL_STATES];
        for (uint32_t i = 0; i < NUM_FINAL_STATES; i++) {
            float bestScore = output[i * ACTIONS];
            for (uint8_t a = 1; a < ACTIONS; a++) {
                if (output[i * ACTIONS + a] > bestScore) {
                    bestScore = output[i * ACTIONS + a];
                }
            }
            nextBestScore[i] = bestScore;
        }
        
        // if (epoch % 64 == 0) {
        //     float maxScore = 0;
        //     float minScore = 0;
        //     float avgScore = 0;
        //     for (uint32_t i = 0; i < NUM_FINAL_STATES; i++) {
        //         if (output[i * ACTIONS + actions[i]] > maxScore) {
        //             maxScore = output[i * ACTIONS + actions[i]];
        //         }
        //         if (output[i * ACTIONS + actions[i]] < minScore) {
        //             minScore = output[i * ACTIONS + actions[i]];
        //         }
        //         avgScore += output[i * ACTIONS + actions[i]];
        //     }
        //     avgScore /= NUM_FINAL_STATES;
        //     printf("Max: %f, Min: %f, Avg: %f, %d/%d\n", maxScore, minScore, avgScore, epoch, EPOCHES);
        // }
        
        // feed states into network
        for (uint32_t i = 0; i < NUM_FINAL_STATES; i++) {
            cudaMemcpy(net.outputs[0] + i * (BOARD_SIZE + 1), states + i * BOARD_SIZE, BOARD_SIZE * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(net.outputs[0] + i * (BOARD_SIZE + 1) + BOARD_SIZE, &one, sizeof(float), cudaMemcpyHostToDevice);
        }
        forwardPropagate(&handle, &net);
        cudaMemcpy(output, net.outputs[net.layers], NUM_FINAL_STATES * ACTIONS * sizeof(float), cudaMemcpyDeviceToHost);
        
        float outputGrad[NUM_FINAL_STATES * ACTIONS]{};
        for (uint32_t i = 0; i < NUM_FINAL_STATES; i++) {
            outputGrad[i * ACTIONS + actions[i]] = rewards[i] + DECAY * nextBestScore[i] - output[i * ACTIONS + actions[i]];
        }
        
        // backpropagate
        cudaMemcpy(net.outputGrad[net.layers], outputGrad, NUM_FINAL_STATES * ACTIONS * sizeof(float), cudaMemcpyHostToDevice);
        backwardPropagate(&handle, &net);
        
        getKeyboardInput(&keyboard);
        uint8_t action = 4;
        board[cy * BOARD_WIDTH + cx] = 0;
        for (uint8_t i = 0; i < keyboard.retBufLen && action == 4; i++) {
            // printf("%d\n", keyboard.buffer[i]);
            switch (keyboard.buffer[i]) {
                case 97: action = 0; if (cx > 0) cx--; break;
                case 100: action = 1; if (cx < BOARD_WIDTH - 1) cx++; break;
                case 119: action = 2; if (cy > 0) cy--; break;
                case 115: action = 3; if (cy < BOARD_WIDTH - 1) cy++; break;
            }
        }
        board[cy * BOARD_WIDTH + cx] = -1;
        uint32_t i = 0;
        for (uint8_t dy = 0; dy < BOARD_WIDTH; dy++) {
            for (uint8_t dx = 0; dx < BOARD_WIDTH; dx++) {
                if (dx == cx && dy == cy) continue;
                board[dy * BOARD_WIDTH + dx] = 1;
                cudaMemcpy(net.outputs[0] + i * (BOARD_SIZE + 1), board, BOARD_SIZE * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(net.outputs[0] + i * (BOARD_SIZE + 1) + BOARD_SIZE, &one, sizeof(float), cudaMemcpyHostToDevice);
                i++;
                board[dy * BOARD_WIDTH + dx] = 0;
            }
        }
        // void printTensor(float* tensor, uint32_t width, uint32_t height) {
        // printTensor(net.outputs[0], (BOARD_SIZE + 1), (BOARD_SIZE - 1));
        net.batchSize = BOARD_SIZE - 1;
        forwardPropagate(&handle, &net);
        cudaMemcpy(output, net.outputs[net.layers], (BOARD_SIZE - 1) * ACTIONS * sizeof(float), cudaMemcpyDeviceToHost);
        net.batchSize = NUM_FINAL_STATES;
        printf("\033[H\033[J");
        i = 0;
        for (uint8_t dy = 0; dy < BOARD_WIDTH; dy++) {
            for (uint8_t dx = 0; dx < BOARD_WIDTH; dx++) {
                switch ((int)board[dy * BOARD_WIDTH + dx]) {
                    case 1: printf("||"); break;
                    case -1: printf("$$"); break;
                    default: printf("..");
                }
                i++;
            }
            printf("\n");
        }
    }
    
    // printParams(&net);
    // printBackParams(&net);
    
    // net.batchSize = 1;
    // uint32_t score = 0;
    // uint32_t steps = 0;
    // memset(board, 0, BOARD_SIZE * sizeof(float));
    // uint8_t px = genNoise(&noise) % BOARD_WIDTH;
    // uint8_t py = genNoise(&noise) % BOARD_WIDTH;
    // board[py * BOARD_WIDTH + px] = 1;
    // uint8_t cx, cy;
    // do {
    //     cx = genNoise(&noise) % BOARD_WIDTH;
    //     cy = genNoise(&noise) % BOARD_WIDTH;
    // } while ((px == cx) && (py == cy));
    // board[cy * BOARD_WIDTH + cx] = -1;
    // while (true) {
    //     printf("\033[H\033[J");
    //     cudaMemcpy(net.outputs[0], board, BOARD_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    //     cudaMemcpy(net.outputs[0] + BOARD_SIZE, &one, sizeof(float), cudaMemcpyHostToDevice);
    //     forwardPropagate(&handle, &net);
    //     float output[ACTIONS];
    //     cudaMemcpy(output, net.outputs[net.layers], ACTIONS * sizeof(float), cudaMemcpyDeviceToHost);
    //     uint8_t bestAction = 0;
    //     for (uint8_t a = 1; a < ACTIONS; a++) {
    //         if (output[a] > output[bestAction]) {
    //             bestAction = a;
    //         }
    //     }
        
    //     // print output and best action
    //     for (uint8_t a = 0; a < ACTIONS; a++) {
    //         printf("%f ", output[a]);
    //     }
    //     printf("\n");
    //     printf("Best action: %d\n", bestAction);
        
    //     board[py * BOARD_WIDTH + px] = 0;
    //     switch (bestAction) {
    //         case 0: if (px > 0) px--; break;
    //         case 1: if (px < BOARD_WIDTH - 1) px++; break;
    //         case 2: if (py > 0) py--; break;
    //         case 3: if (py < BOARD_WIDTH - 1) py++; break;
    //     }
    //     board[py * BOARD_WIDTH + px] = 1;
    //     score += (px == cx) && (py == cy);
    //     steps++;
    //     while ((px == cx) && (py == cy)) {
    //         cx = genNoise(&noise) % BOARD_WIDTH;
    //         cy = genNoise(&noise) % BOARD_WIDTH;
    //     }
    //     board[cy * BOARD_WIDTH + cx] = -1;
        
    //     for (uint8_t dy = 0; dy < BOARD_WIDTH; dy++) {
    //         for (uint8_t dx = 0; dx < BOARD_WIDTH; dx++) {
    //             switch ((int)board[dy * BOARD_WIDTH + dx]) {
    //                 case 1: printf("||"); break;
    //                 case -1: printf("$$"); break;
    //                 default: printf("..");
    //             }
    //         }
    //         printf("\n");
    //     }
    //     printf("Score: %d\n", score);
    //     printf("Steps: %d\n", steps);
        
    //     struct timeval tv;
    //     tv.tv_sec = 0;
    //     tv.tv_usec = 200000;
    //     select(0, NULL, NULL, NULL, &tv);
    // }

    return 0;
}