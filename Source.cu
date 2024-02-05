#include <stdio.h>
#include <stdint.h>
#include <sys/time.h>

#include "Header.cuh"


inline void checkCudaStatus(cudaError_t status) {
    if (status != cudaSuccess) {
        printf("cuda API failed with status %d: %s\n", status, cudaGetErrorString(status));
        exit(-1);
    }
}

void mixSeed(uint32_t *seed1, uint32_t *seed2) {
    *seed2 *= 0xbf324c81;
    *seed1 ^= *seed2 ^ 0x4ba1bb47;
    *seed1 *= 0x9c7493ad;
    *seed2 ^= *seed1 ^ 0xb7ebcb79;
}

void fillSeeds(uint32_t *seed1, uint32_t *seed2) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    *seed1 = tv.tv_sec;
    *seed2 = tv.tv_usec;
    for (uint8_t i = 8; i--;) mixSeed(seed1, seed2);
}

void fillHData(uint8_t *hData, const uint32_t seed1, const uint32_t seed2) {
    uint8_t *dPerm;
    uint8_t *dData;
    
    checkCudaStatus(cudaMalloc((void**)&dPerm, 0x400 * sizeof(uint8_t)));
    checkCudaStatus(cudaMalloc((void**)&dData, 0x100000000 * sizeof(uint8_t)));
    
    fillDPerm<<<1, 0x400>>>(dPerm, seed1, seed2);
    fillDData<<<0x400000, 0x400>>>(dData, dPerm, 1, 2048, 8, 0.5);
    
    checkCudaStatus(cudaMemcpy(hData, dData, 0x100000000 * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    
    checkCudaStatus(cudaFree(dPerm));
    checkCudaStatus(cudaFree(dData));
}

int main() {
    uint32_t seed1, seed2;
    fillSeeds(&seed1, &seed2);
    
    uint8_t *hData = (uint8_t*)malloc(0x100000000 * sizeof(uint8_t));
    fillHData(hData, seed1, seed2);
    
    const uint8_t VIEW_RADIUS = 16;
    const uint8_t VIEW_SIZE = VIEW_RADIUS * 2 + 1;
    
    uint16_t x = 0, y = 0;
    uint8_t move;
    
    while(1) {
        system("clear");
        for (uint16_t i = VIEW_SIZE, ry = y + VIEW_RADIUS; i--; ry--) {
            for (uint16_t j = VIEW_SIZE, rx = x + VIEW_RADIUS; j--; rx--) {
                switch (hData[(uint32_t)ry << 16 | rx]) {
                    case 0: printf("\x1b[38;2;040;150;160m..\x1b[0m"); break;
                    case 1: printf("\x1b[38;2;050;190;170m--\x1b[0m"); break;
                    case 2: printf("\x1b[38;2;140;210;210m;;\x1b[0m"); break;
                    case 3: printf("\x1b[38;2;230;220;210m==\x1b[0m"); break;
                    case 4: printf("\x1b[38;2;200;170;140m**\x1b[0m"); break;
                    case 5: printf("\x1b[38;2;090;190;090m++\x1b[0m"); break;
                    case 6: printf("\x1b[38;2;040;120;040m##\x1b[0m"); break;
                    case 7: printf("\x1b[38;2;000;060;010m@@\x1b[0m"); break;
                }
            }
            printf("\n");
        }

        printf("Move (wasd): ");
        scanf(" %c", &move);

        x += ((move == 'a') - (move == 'd')) * 16;
        y += ((move == 'w') - (move == 's')) * 16;
    }
    
    free(hData);
    return 0;
}