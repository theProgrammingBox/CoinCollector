#include <stdio.h>
#include <stdint.h>
#include <sys/time.h>

#include "Header2.cuh"

inline void checkCudaStatus(cudaError_t status) {
    if (status != cudaSuccess) {
        printf("cuda API failed with status %d: %s\n", status, cudaGetErrorString(status));
        exit(-1);
    }
}

int main() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    uint32_t seed1 = tv.tv_sec ^ 0xd083b1c1;
    uint32_t seed2 = tv.tv_usec ^ 0xae1233fd;
    for (int i = 8; i--;) {
        seed2 *= 0xbf324c81;
        seed1 ^= seed2;
        seed1 *= 0x9c7493ad;
        seed2 ^= seed1;
    }
    
    uint16_t *dPerm;
    uint8_t *dData;
    uint8_t *hData = (uint8_t*)malloc(0x100000000 * sizeof(uint8_t));
    
    checkCudaStatus(cudaMalloc((void**)&dPerm, 0x400 * sizeof(uint16_t)));
    checkCudaStatus(cudaMalloc((void**)&dData, 0x100000000 * sizeof(uint8_t)));
    
    fillDPerm<<<1, 0x400>>>(dPerm, seed1, seed2);
    fillDData<<<0x400000, 0x400>>>(dData, dPerm, 1000);
    
    checkCudaStatus(cudaMemcpy(hData, dData, 0x100000000 * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    
    checkCudaStatus(cudaFree(dPerm));
    checkCudaStatus(cudaFree(dData));
    
    
    
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
                    case 6: printf("\x1b[38;2;040;140;080m##\x1b[0m"); break;
                    case 7: printf("\x1b[38;2;000;080;030m@@\x1b[0m"); break;
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