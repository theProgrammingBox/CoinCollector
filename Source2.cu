#include <stdio.h>
#include <stdint.h>
#include <sys/time.h>

#include "Header2.cuh"

void printD16(const uint16_t *dData, uint32_t size) {
    uint16_t *hData = (uint16_t*)malloc(size * sizeof(uint16_t));
    checkCudaStatus(cudaMemcpy(hData, dData, size * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    for (uint32_t i = 0; i < size; i++) printf("%d ", hData[i]);
    printf("\n\n");
    free(hData);
}

void printD8(const uint8_t *dData, uint32_t size) {
    uint8_t *hData = (uint8_t*)malloc(size * sizeof(uint8_t));
    checkCudaStatus(cudaMemcpy(hData, dData, size * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    for (uint32_t i = 0; i < size; i++) printf("%d ", hData[i]);
    printf("\n\n");
    free(hData);
}

int main() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    uint32_t seed = tv.tv_sec ^ tv.tv_usec ^ 0x85f35457;
    for (int i = 8; i--;) {
        seed *= 0xBAC57D37;
        seed ^= seed << 13;
        seed *= 0x24F66AC9;
        seed ^= seed >> 17;
    }
    
    uint16_t *dPerm;
    uint8_t *dData;
    uint8_t *hData = (uint8_t*)malloc(0x100000000 * sizeof(uint8_t));
    
    checkCudaStatus(cudaMalloc((void**)&dPerm, 0x100 * sizeof(uint16_t)));
    checkCudaStatus(cudaMalloc((void**)&dData, 0x100000000 * sizeof(uint8_t)));
    
    fillDPerm<<<1, 0x100>>>(dPerm, seed);
    fillDData<<<0x400000, 0x400>>>(dData, dPerm);
    
    checkCudaStatus(cudaMemcpy(hData, dData, 0x100000000 * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    
    printD16(dPerm, 0x100);
    printD8(dData, 0x100);
    
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
                switch (hData[ry << 16 | rx]) {
                    case 0: printf("\x1b[38;2;000;000;139m..\x1b[0m"); break;
                    case 1: printf("\x1b[38;2;000;105;148m--\x1b[0m"); break;
                    case 2: printf("\x1b[38;2;173;216;230m;;\x1b[0m"); break;
                    case 3: printf("\x1b[38;2;194;178;128m==\x1b[0m"); break;
                    case 4: printf("\x1b[38;2;155;118;083m**\x1b[0m"); break;
                    case 5: printf("\x1b[38;2;100;200;100m++\x1b[0m"); break;
                    case 6: printf("\x1b[38;2;010;120;010m##\x1b[0m"); break;
                    case 7: printf("\x1b[38;2;000;080;000m@@\x1b[0m"); break;
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