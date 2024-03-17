#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/time.h>

#include <cublas_v2.h>

#define BOARD_WIDTH 3
#define BOARD_SIZE (BOARD_WIDTH * BOARD_WIDTH)
#define ACTIONS 4
#define NUM_FINAL_STATES (BOARD_SIZE * (BOARD_SIZE - 1) * ACTIONS)

#define HIDDEN_LAYER_SIZE 16
#define DECAY 0.9

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

struct Model {
    float* weight1;//[BOARD_SIZE * HIDDEN_LAYER_SIZE];
    float* weight2;//[HIDDEN_LAYER_SIZE * ACTIONS];
    float* bias1;//[HIDDEN_LAYER_SIZE];
    float* bias2;//[ACTIONS];
    
    float* input;//[BOARD_SIZE * NUM_FINAL_STATES];
    float* hidden;//[HIDDEN_LAYER_SIZE * NUM_FINAL_STATES];
    float* output;//[ACTIONS * NUM_FINAL_STATES];
    
    float* weight1Grad;//[BOARD_SIZE * HIDDEN_LAYER_SIZE];
    float* weight2Grad;//[HIDDEN_LAYER_SIZE * ACTIONS];
    
    float* hiddenGrad;//[HIDDEN_LAYER_SIZE * NUM_FINAL_STATES];
    float* outputGrad;//[BOARD_SIZE * NUM_FINAL_STATES];
};

void initializeModel(Model *model, uint32_t* seed1, uint32_t* seed2) {
    cudaMalloc((void**)&model->weight1, BOARD_SIZE * HIDDEN_LAYER_SIZE * sizeof(float));
    cudaMalloc((void**)&model->weight2, HIDDEN_LAYER_SIZE * ACTIONS * sizeof(float));
    cudaMalloc((void**)&model->bias1, HIDDEN_LAYER_SIZE * sizeof(float));
    cudaMalloc((void**)&model->bias2, ACTIONS * sizeof(float));
    
    cudaMalloc((void**)&model->input, BOARD_SIZE * NUM_FINAL_STATES * sizeof(float));
    cudaMalloc((void**)&model->hidden, HIDDEN_LAYER_SIZE * NUM_FINAL_STATES * sizeof(float));
    cudaMalloc((void**)&model->output, ACTIONS * NUM_FINAL_STATES * sizeof(float));
    
    cudaMalloc((void**)&model->weight1Grad, BOARD_SIZE * HIDDEN_LAYER_SIZE * sizeof(float));
    cudaMalloc((void**)&model->weight2Grad, HIDDEN_LAYER_SIZE * ACTIONS * sizeof(float));
    
    cudaMalloc((void**)&model->hiddenGrad, HIDDEN_LAYER_SIZE * NUM_FINAL_STATES * sizeof(float));
    cudaMalloc((void**)&model->outputGrad, BOARD_SIZE * NUM_FINAL_STATES * sizeof(float));
    
    fillUniform(model->weight1, BOARD_SIZE * HIDDEN_LAYER_SIZE, seed1, seed2, -0.1, 0.1);
    fillUniform(model->weight2, HIDDEN_LAYER_SIZE * ACTIONS, seed1, seed2, -0.1, 0.1);
    fillUniform(model->bias1, HIDDEN_LAYER_SIZE, seed1, seed2, -0.1, 0.1);
    fillUniform(model->bias2, ACTIONS, seed1, seed2, -0.1, 0.1);
}

void forward(cublasHandle_t* handle, Model *model) {
    const float ONE = 1;
    
    for (uint32_t i = 0; i < NUM_FINAL_STATES; i++) {
        cudaMemcpy(model->hidden + i * HIDDEN_LAYER_SIZE, model->bias1, HIDDEN_LAYER_SIZE * sizeof(float), cudaMemcpyDeviceToDevice);
    }
    
    cublasSgemm(
        *handle, CUBLAS_OP_N, CUBLAS_OP_N,
        HIDDEN_LAYER_SIZE, NUM_FINAL_STATES, BOARD_SIZE,
        &ONE,
        model->weight1, HIDDEN_LAYER_SIZE,
        model->input, BOARD_SIZE,
        &ONE,
        model->hidden, HIDDEN_LAYER_SIZE
    );
    
    reluForward(model->hidden, HIDDEN_LAYER_SIZE * NUM_FINAL_STATES);
    for (uint32_t i = 0; i < NUM_FINAL_STATES; i++) {
        cudaMemcpy(model->output + i * ACTIONS, model->bias2, ACTIONS * sizeof(float), cudaMemcpyDeviceToDevice);
    }
    cublasSgemm(
        *handle, CUBLAS_OP_N, CUBLAS_OP_N,
        ACTIONS, NUM_FINAL_STATES, HIDDEN_LAYER_SIZE,
        &ONE,
        model->weight2, ACTIONS,
        model->hidden, HIDDEN_LAYER_SIZE,
        &ONE,
        model->output, ACTIONS
    );
}

void backward(cublasHandle_t* handle, Model *model) {
    const float learingRate = 0.00016;
    const float ONE = 1;
    const float ZERO = 0;
    
    cublasSgemm(
        *handle, CUBLAS_OP_N, CUBLAS_OP_T,
        ACTIONS, HIDDEN_LAYER_SIZE, NUM_FINAL_STATES,
        &ONE,
        model->outputGrad, ACTIONS,
        model->hidden, HIDDEN_LAYER_SIZE,
        &ZERO,
        model->weight2Grad, ACTIONS
    );
    
    cublasSgemm(
        *handle, CUBLAS_OP_T, CUBLAS_OP_N,
        HIDDEN_LAYER_SIZE, NUM_FINAL_STATES, ACTIONS,
        &ONE,
        model->weight2, ACTIONS,
        model->outputGrad, ACTIONS,
        &ZERO,
        model->hiddenGrad, HIDDEN_LAYER_SIZE
    );
    
    reluBackward(model->hidden, model->hiddenGrad, HIDDEN_LAYER_SIZE * NUM_FINAL_STATES);
    
    cublasSgemm(
        *handle, CUBLAS_OP_N, CUBLAS_OP_T,
        HIDDEN_LAYER_SIZE, BOARD_SIZE, NUM_FINAL_STATES,
        &ONE,
        model->hiddenGrad, HIDDEN_LAYER_SIZE,
        model->input, BOARD_SIZE,
        &ZERO,
        model->weight1Grad, HIDDEN_LAYER_SIZE
    );
    
    cublasSaxpy(*handle, HIDDEN_LAYER_SIZE * ACTIONS, &learingRate, model->weight2Grad, 1, model->weight2, 1);
    cublasSaxpy(*handle, BOARD_SIZE * HIDDEN_LAYER_SIZE, &learingRate, model->weight1Grad, 1, model->weight1, 1);
    cublasSaxpy(*handle, ACTIONS, &learingRate, model->outputGrad, 1, model->bias2, 1);
    cublasSaxpy(*handle, HIDDEN_LAYER_SIZE, &learingRate, model->hiddenGrad, 1, model->bias1, 1);
}

void copyParams(Model *model, Model *frozenModel) {
    cudaMemcpy(frozenModel->weight1, model->weight1, BOARD_SIZE * HIDDEN_LAYER_SIZE * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(frozenModel->weight2, model->weight2, HIDDEN_LAYER_SIZE * ACTIONS * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(frozenModel->bias1, model->bias1, HIDDEN_LAYER_SIZE * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(frozenModel->bias2, model->bias2, ACTIONS * sizeof(float), cudaMemcpyDeviceToDevice);
}

int main(int argc, char *argv[])
{
    uint32_t seed1, seed2;
    initializeSeeds(&seed1, &seed2);
    
    float states[BOARD_SIZE * NUM_FINAL_STATES];
    uint8_t actions[NUM_FINAL_STATES];
    float rewards[NUM_FINAL_STATES];
    float nextStates[BOARD_SIZE * NUM_FINAL_STATES];
    
    uint32_t queueIdx = 0;
    float board[9]{};
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
                            cxx = mixSeed(&seed1, &seed2) % BOARD_WIDTH;
                            cyy = mixSeed(&seed1, &seed2) % BOARD_WIDTH;
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
                    board[py * 3 + px] = 0;
                    board[cy * 3 + cx] = 0;
                }
            }
        }
    }
    
    // print states
    for (uint32_t i = 0; i < NUM_FINAL_STATES; i++) {
        for (uint8_t dy = 0; dy < BOARD_WIDTH; dy++) {
            for (uint8_t dx = 0; dx < BOARD_WIDTH; dx++) {
                printf("%.0f ", states[i * BOARD_SIZE + dy * BOARD_WIDTH + dx]);
            }
            printf("\n");
        }
        printf("Action: %d\n", actions[i]);
        printf("Reward: %.0f\n", rewards[i]);
        for (uint8_t dy = 0; dy < BOARD_WIDTH; dy++) {
            for (uint8_t dx = 0; dx < BOARD_WIDTH; dx++) {
                printf("%.0f ", nextStates[i * BOARD_SIZE + dy * BOARD_WIDTH + dx]);
            }
            printf("\n");
        }
        printf("\n");
    }
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    Model model;
    Model frozenModel;
    initializeModel(&model, &seed1, &seed2);
    initializeModel(&frozenModel, &seed1, &seed2);
    
    for (uint32_t epoch = 0; epoch < (1 << 10); epoch++) {
        if (epoch % 8 == 0) {
            copyParams(&model, &frozenModel);
        }
        
        // outputGrad = rewards + DECAY * nextBestScore - output
        cudaMemcpy(frozenModel.input, nextStates, BOARD_SIZE * NUM_FINAL_STATES * sizeof(float), cudaMemcpyHostToDevice);
        forward(&handle, &frozenModel);
        float output[NUM_FINAL_STATES * ACTIONS];
        cudaMemcpy(output, frozenModel.output, NUM_FINAL_STATES * ACTIONS * sizeof(float), cudaMemcpyDeviceToHost);
        float nextBestScore[NUM_FINAL_STATES];
        for (uint32_t i = 0; i < NUM_FINAL_STATES; i++) {
            nextBestScore[i] = output[i * ACTIONS];
            for (uint8_t a = 1; a < ACTIONS; a++) {
                if (output[i * ACTIONS + a] > nextBestScore[i]) {
                    nextBestScore[i] = output[i * ACTIONS + a];
                }
            }
        }
        float outputGrad[NUM_FINAL_STATES]{};
        for (uint32_t i = 0; i < NUM_FINAL_STATES; i++) {
            outputGrad[i * ACTIONS + actions[i]] = rewards[i] + DECAY * nextBestScore[i] - output[i * ACTIONS + actions[i]];
        }
        
        float maxScore = 0;
        float minScore = 0;
        float avgScore = 0;
        for (uint32_t i = 0; i < NUM_FINAL_STATES; i++) {
            if (output[i * ACTIONS + actions[i]] > maxScore) {
                maxScore = output[i * ACTIONS + actions[i]];
            }
            if (output[i * ACTIONS + actions[i]] < minScore) {
                minScore = output[i * ACTIONS + actions[i]];
            }
            avgScore += output[i * ACTIONS + actions[i]];
        }
        avgScore /= NUM_FINAL_STATES;
        printf("Max: %f, Min: %f, Avg: %f\n", maxScore, minScore, avgScore);
        
        cudaMemcpy(model.input, states, BOARD_SIZE * NUM_FINAL_STATES * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(model.outputGrad, outputGrad, NUM_FINAL_STATES * ACTIONS * sizeof(float), cudaMemcpyHostToDevice);
        forward(&handle, &model);
        backward(&handle, &model);
    }

    return 0;
}