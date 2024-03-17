#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/time.h>

// #include <cuda_runtime.h>
#include <cublas_v2.h>

#define LEARNING_RATE 0.01
#define DISCOUNT_FACTOR 0.9
#define BOARD_WIDTH 2
#define EPOCHS 168
#define QUEUE_LENGTH 1024
#define MAX_BATCH_SIZE 64
#define HIDDEN_LAYER_SIZE 16
#define ACTIONS 4
#define BOARD_SIZE (BOARD_WIDTH * BOARD_WIDTH)

/*
TODO:
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

__global__ void _add(float* arr, float* arrGrad, float scalar, float* elemMulArr2, uint32_t size) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        if (elemMulArr2 != NULL) arr[index] += arrGrad[index] * scalar * elemMulArr2[index] - arr[index] * 0.1f;
        else arr[index] += arrGrad[index] * scalar - arr[index] * 0.1f;
    }
}

void add(float* arr, float* arrGrad, float scalar, float* elemMulArr2, uint32_t size) {
    _add<<<(size >> 10) + (size & 0x3ff ? 1 : 0), 1024>>>(arr, arrGrad, scalar, elemMulArr2, size);
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
    
    cudaMalloc((void**)&model->weight1Grad, BOARD_SIZE * HIDDEN_LAYER_SIZE * sizeof(float));
    cudaMalloc((void**)&model->weight2Grad, HIDDEN_LAYER_SIZE * ACTIONS * sizeof(float));
    
    cudaMalloc((void**)&model->hiddenGrad, HIDDEN_LAYER_SIZE * MAX_BATCH_SIZE * sizeof(float));
    cudaMalloc((void**)&model->outputGrad, BOARD_SIZE * MAX_BATCH_SIZE * sizeof(float));
    
    fillUniform(model->weight1, BOARD_SIZE * HIDDEN_LAYER_SIZE, seed1, seed2, -0.1, 0.1);
    fillUniform(model->weight2, HIDDEN_LAYER_SIZE * ACTIONS, seed1, seed2, -0.1, 0.1);
    fillUniform(model->bias1, HIDDEN_LAYER_SIZE, seed1, seed2, -0.1, 0.1);
    fillUniform(model->bias2, ACTIONS, seed1, seed2, -0.1, 0.1);
    
    fill(model->weight1Var, BOARD_SIZE * HIDDEN_LAYER_SIZE, 0.1);
    fill(model->weight2Var, HIDDEN_LAYER_SIZE * ACTIONS, 0.1);
    fill(model->bias1Var, HIDDEN_LAYER_SIZE, 0.1);
    fill(model->bias2Var, ACTIONS, 0.1);
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

void forward(cublasHandle_t* handle, uint32_t batchSize, Model *model, uint8_t noise, uint32_t* seed1, uint32_t* seed2) {
    const float ONE = 1;
    const float ZERO = 0;
    
    float* weight1, *weight2, *bias1, *bias2;
    if (noise == 1) {
        newNoise(model, seed1, seed2);
        weight1 = model->newWeight1;
        weight2 = model->newWeight2;
        bias1 = model->newBias1;
        bias2 = model->newBias2;
    } else {
        weight1 = model->weight1;
        weight2 = model->weight2;
        bias1 = model->bias1;
        bias2 = model->bias2;
    }
    
    // if (noise == 0) {
    // //print bias1
    // printf("Bias1:\n");
    // printTensor(bias1, 1, HIDDEN_LAYER_SIZE);
    // }
    
    for (uint32_t i = 0; i < batchSize; i++) {
        cudaMemcpy(model->hidden + i * HIDDEN_LAYER_SIZE, bias1, HIDDEN_LAYER_SIZE * sizeof(float), cudaMemcpyDeviceToDevice);
    }
    
    // if (noise == 0) {
    // // print hidden
    // printf("Hidden:\n");
    // printTensor(model->hidden, batchSize, HIDDEN_LAYER_SIZE);
    // }
    
    cublasSgemm(
        *handle, CUBLAS_OP_N, CUBLAS_OP_N,
        HIDDEN_LAYER_SIZE, batchSize, BOARD_SIZE,
        &ONE,
        weight1, HIDDEN_LAYER_SIZE,
        model->input, BOARD_SIZE,
        &ONE,
        model->hidden, HIDDEN_LAYER_SIZE
    );
    
    reluForward(model->hidden, HIDDEN_LAYER_SIZE * batchSize);
    for (uint32_t i = 0; i < batchSize; i++) {
        cudaMemcpy(model->output + i * ACTIONS, bias2, ACTIONS * sizeof(float), cudaMemcpyDeviceToDevice);
    }
    cublasSgemm(
        *handle, CUBLAS_OP_N, CUBLAS_OP_N,
        ACTIONS, batchSize, HIDDEN_LAYER_SIZE,
        &ONE,
        weight2, ACTIONS,
        model->hidden, HIDDEN_LAYER_SIZE,
        &ONE,
        model->output, ACTIONS
    );
}

void backward(cublasHandle_t* handle, uint32_t batchSize, Model *model) {
    const float ONE = 1;
    const float ZERO = 0;
    
    cublasSgemm(
        *handle, CUBLAS_OP_N, CUBLAS_OP_T,
        ACTIONS, HIDDEN_LAYER_SIZE, batchSize,
        &ONE,
        model->outputGrad, ACTIONS,
        model->hidden, HIDDEN_LAYER_SIZE,
        &ZERO,
        model->weight2Grad, ACTIONS
    );
    // // print weight2Grad
    // printf("Weight2Grad:\n");
    // printTensor(model->weight2Grad, HIDDEN_LAYER_SIZE, ACTIONS);
    
    
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
    
    cublasSgemm(
        *handle, CUBLAS_OP_N, CUBLAS_OP_T,
        HIDDEN_LAYER_SIZE, BOARD_SIZE, batchSize,
        &ONE,
        model->hiddenGrad, HIDDEN_LAYER_SIZE,
        model->input, BOARD_SIZE,
        &ZERO,
        model->weight1Grad, HIDDEN_LAYER_SIZE
    );
    
    add(model->bias2, model->outputGrad, LEARNING_RATE, NULL, ACTIONS * batchSize);
    add(model->bias2Var, model->outputGrad, LEARNING_RATE, model->bias2Sample, ACTIONS * batchSize);
    add(model->weight2, model->weight2Grad, LEARNING_RATE, NULL, HIDDEN_LAYER_SIZE * ACTIONS);
    add(model->weight2Var, model->weight2Grad, LEARNING_RATE, model->weight2Sample, HIDDEN_LAYER_SIZE * ACTIONS);
    add(model->bias1, model->hiddenGrad, LEARNING_RATE, NULL, HIDDEN_LAYER_SIZE * batchSize);
    add(model->bias1Var, model->hiddenGrad, LEARNING_RATE, model->bias1Sample, HIDDEN_LAYER_SIZE * batchSize);
    add(model->weight1, model->weight1Grad, LEARNING_RATE, NULL, BOARD_SIZE * HIDDEN_LAYER_SIZE);
    add(model->weight1Var, model->weight1Grad, LEARNING_RATE, model->weight1Sample, BOARD_SIZE * HIDDEN_LAYER_SIZE);
}

void printParams(Model *model) {
    printTensor(model->weight1, BOARD_SIZE, HIDDEN_LAYER_SIZE);
    printTensor(model->bias1, 1, HIDDEN_LAYER_SIZE);
    printTensor(model->weight2, HIDDEN_LAYER_SIZE, ACTIONS);
    printTensor(model->bias2, 1, ACTIONS);
}

void copyParams(Model *model, Model *frozenModel) {
    cudaMemcpy(frozenModel->weight1, model->weight1, BOARD_SIZE * HIDDEN_LAYER_SIZE * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(frozenModel->weight2, model->weight2, HIDDEN_LAYER_SIZE * ACTIONS * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(frozenModel->bias1, model->bias1, HIDDEN_LAYER_SIZE * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(frozenModel->bias2, model->bias2, ACTIONS * sizeof(float), cudaMemcpyDeviceToDevice);
}

int main(int argc, char *argv[])
{
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    // for random number generation
    uint32_t seed1, seed2;
    initializeSeeds(&seed1, &seed2);
    
    Model model, frozenModel;
    initializeModel(&model, &seed1, &seed2);
    initializeModel(&frozenModel, &seed1, &seed2);
    
    uint32_t queueIndex = 0;
    float queueInitState[QUEUE_LENGTH * BOARD_SIZE]{};
    float queueResState[QUEUE_LENGTH * BOARD_SIZE]{};
    uint8_t queueAction[QUEUE_LENGTH]{};
    float queueReward[QUEUE_LENGTH]{};
    
    float actions[ACTIONS];
    uint32_t sampleIndex[MAX_BATCH_SIZE];
    float outputScores[MAX_BATCH_SIZE * ACTIONS];
    float bestNextScores[MAX_BATCH_SIZE];
    
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
        cudaMemcpy(model.input, board, BOARD_SIZE * sizeof(float), cudaMemcpyHostToDevice);
        forward(&handle, 1, &model, 1, &seed1, &seed2);
        cudaMemcpy(actions, model.output, ACTIONS * sizeof(float), cudaMemcpyDeviceToHost);
        action = 0;
        for (uint8_t i = 1; i < ACTIONS; i++) {
            if (actions[i] > actions[action]) action = i;
        }
        
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
        uint32_t tmp;
        printf("Epoch: %d\n", epoch);
        for (tmp = 0; tmp < batchSize; tmp++) {
            sampleIndex[tmp] = mixSeed(&seed1, &seed2) % queueUpperIndex;
        }
        
        //  compute bestNextScores, fill in batch and forward propogate batch with no noise
        for (tmp = 0; tmp < batchSize; tmp++) {
            cudaMemcpy(frozenModel.input + tmp * BOARD_SIZE, queueInitState + sampleIndex[tmp] * BOARD_SIZE, BOARD_SIZE * sizeof(float), cudaMemcpyHostToDevice);
        }
        
        printf("Batch input:\n");
        printTensor(frozenModel.input, batchSize, BOARD_SIZE);
        
        if (epoch == EPOCHS - 1) {
            printf("weight1:\n");
            printTensor(frozenModel.weight1, BOARD_SIZE, HIDDEN_LAYER_SIZE);
            printf("bias1:\n");
            printTensor(frozenModel.bias1, 1, HIDDEN_LAYER_SIZE);
            printf("hidden:\n");
            printTensor(frozenModel.hidden, batchSize, HIDDEN_LAYER_SIZE);
            printf("weight2:\n");
            printTensor(frozenModel.weight2, HIDDEN_LAYER_SIZE, ACTIONS);
            printf("bias2:\n");
            printTensor(frozenModel.bias2, 1, ACTIONS);
        }
        
        // unfreeze model every 16 epochs to prevent overestimation
        if (epoch % 64 == 0) {
            copyParams(&model, &frozenModel);
        }
        forward(&handle, batchSize, &frozenModel, 0, NULL, NULL);
        cudaMemcpy(outputScores, frozenModel.output, batchSize * ACTIONS * sizeof(float), cudaMemcpyDeviceToHost);
        
        printf("output (Output scores):\n");
        printTensor(frozenModel.output, batchSize, ACTIONS);
        
        for (tmp = 0; tmp < batchSize; tmp++) {
            bestNextScores[tmp] = outputScores[tmp * ACTIONS];
            for (uint8_t i = 1; i < ACTIONS; i++) {
                if (outputScores[tmp * ACTIONS + i] > bestNextScores[tmp]) bestNextScores[tmp] = outputScores[tmp * ACTIONS + i];
            }
        }
        
        // calculating outputGrad, setting the batch input to the initial state, and forward propogating with noise
        for (tmp = 0; tmp < batchSize; tmp++) {
            cudaMemcpy(model.input + tmp * BOARD_SIZE, queueInitState + sampleIndex[tmp] * BOARD_SIZE, BOARD_SIZE * sizeof(float), cudaMemcpyHostToDevice);
        }
        forward(&handle, batchSize, &model, 1, &seed1, &seed2);
        // memset gradiant to 0
        cudaMemset(model.outputGrad, 0, batchSize * ACTIONS * sizeof(float));
        float outputGrad[batchSize * ACTIONS]{};
        for (tmp = 0; tmp < batchSize; tmp++) {
            // outputGrad = output - (reward + discountFactor * bestNextScore)
            outputGrad[tmp * ACTIONS + queueAction[sampleIndex[tmp]]] =  queueReward[sampleIndex[tmp]] + DISCOUNT_FACTOR * bestNextScores[tmp] - outputScores[tmp * ACTIONS + queueAction[sampleIndex[tmp]]];
            // print outputScores, reward, bestNextScores, and outputGrad
            printf("outputGrad = %f - (%f + %f * %f) = %f\n", outputScores[tmp * ACTIONS + queueAction[sampleIndex[tmp]]], queueReward[sampleIndex[tmp]], DISCOUNT_FACTOR, bestNextScores[tmp], outputGrad[tmp * ACTIONS + queueAction[sampleIndex[tmp]]]);
        }
        printf("\n");
        cudaMemcpy(model.outputGrad, outputGrad, batchSize * ACTIONS * sizeof(float), cudaMemcpyHostToDevice);
        backward(&handle, batchSize, &model);
        
        
            
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
    
    // printParams(&model);
    return 0;
    
    // now run the model forever
    memset(board, 0, BOARD_SIZE * sizeof(float));
    x = mixSeed(&seed1, &seed2) % BOARD_WIDTH;
    y = mixSeed(&seed1, &seed2) % BOARD_WIDTH;
    do {
        cx = mixSeed(&seed1, &seed2) % BOARD_WIDTH;
        cy = mixSeed(&seed1, &seed2) % BOARD_WIDTH;
    } while (x == cx && y == cy);
    board[x + y * BOARD_WIDTH] = 1;
    board[cx + cy * BOARD_WIDTH] = 2;
    while (1) {
        // clear terminal
        printf("\033[H\033[J");
        
        cudaMemcpy(model.input, board, BOARD_SIZE * sizeof(float), cudaMemcpyHostToDevice);
        forward(&handle, 1, &model, 1, &seed1, &seed2);
        cudaMemcpy(actions, model.output, ACTIONS * sizeof(float), cudaMemcpyDeviceToHost);
        action = 0;
        for (uint8_t i = 1; i < ACTIONS; i++) {
            if (actions[i] > actions[action]) action = i;
        }
        
        // apply action
        board[x + y * BOARD_WIDTH] = 0;
        switch (action) {
            case 0: if (x > 0) x--; break;
            case 1: if (x < BOARD_WIDTH - 1) x++; break;
            case 2: if (y > 0) y--; break;
            case 3: if (y < BOARD_WIDTH - 1) y++; break;
        }
        board[x + y * BOARD_WIDTH] = 1;
        
        while (x == cx && y == cy) {
            cx = mixSeed(&seed1, &seed2) % BOARD_WIDTH;
            cy = mixSeed(&seed1, &seed2) % BOARD_WIDTH;
        }
        board[cx + cy * BOARD_WIDTH] = 2;
        
        for (x = 0; x < 2; x++) {
            for (y = 0; y < 2; y++) {
                printf("%.0f ", board[x + y * 2]);
            }
            printf("\n");
        }
        printf("\n");
    }
    
    return 0;
}