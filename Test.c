#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

/*
- Grass
-- has 3 stages, .., ss and SS
*/

uint16_t rand16(uint32_t* seed) {
    *seed ^= *seed >> 16;
    *seed *= 0xBAC57D37;
    *seed ^= *seed >> 16;
    *seed *= 0x24F66AC9;
    return *seed;
}

uint8_t hash8(uint32_t seed, uint16_t idx) {
    seed ^= idx;
    seed *= 0xF7C2EBCD;
    seed ^= seed >> 16;
    seed *= 0xBAC57D37;
    seed ^= seed >> 16;
    seed *= 0x24F66AC9;
    return seed;
}

void placeCoin(uint8_t *grid, uint32_t* seed) {
    uint16_t pos;
    do {
        pos = rand16(seed);
    } while (grid[pos]);
    grid[pos] = 2;
}

int main() {
    uint32_t seed = time(NULL);
    for (int i = 0x10; i--; ) {
        for (int j = 0x10; j--; ) {
            uint16_t pos1 = i << 4 | j;
            uint16_t pos2 = (i & 0xfe) << 4 | (j & 0xfe);
            uint16_t pos3 = (i & 0xfc) << 4 | (j & 0xfc);
            uint8_t a = hash8(seed, pos1) & 1;
            uint8_t b = hash8(seed, pos2) & 1;
            uint8_t c = hash8(seed, pos3) & 1;
            switch (a + b * 2 + c * 3) {
                case 0: printf("  "); break;
                case 1: printf(".."); break;
                case 2: printf(",,"); break;
                case 3: printf("::"); break;
                case 4: printf("ii"); break;
                case 5: printf("ll"); break;
                case 6: printf("WW"); break;
            }
        }
        printf("\n");
    }
    
    // const uint16_t NUM_COINS = 256 * 16;
    // const uint8_t VIEW_RADIUS = 8;
    // const uint8_t VIEW_SIZE = VIEW_RADIUS * 2 + 1;

    // uint32_t seed = time(NULL);
    // uint8_t *grid = malloc(0x10000);
    // uint8_t x = 0, y = 0, move;
    // uint16_t coins = 0, pidx = 0;
    
    // memset(grid, 0, 0x10000);
    // grid[0] = 1;
    // for (int i = 0; i < NUM_COINS; ++i) placeCoin(grid, &seed);
    
    // while(1) {
    //     system("clear");
    //     for (uint8_t i = VIEW_SIZE, ry = y + VIEW_RADIUS; i--; ry--) {
    //         for (uint8_t j = VIEW_SIZE, rx = x + VIEW_RADIUS; j--; rx--) {
    //             switch (grid[ry << 8 | rx]) {
    //                 case 0: printf(".."); break;
    //                 case 1: printf("[]"); break;
    //                 case 2: printf("qp"); break;
    //             }
    //         }
    //         printf("\n");
    //     }
        
    //     printf("Coins: %d\n", coins);
    //     printf("Move (wasd): ");
    //     scanf(" %c", &move);
        
    //     grid[pidx] = 0;
    //     x += (move == 'a') - (move == 'd');
    //     y += (move == 'w') - (move == 's');
    //     pidx = y << 8 | x;
    //     if (grid[pidx] == 2) {
    //         coins++;
    //         placeCoin(grid, &seed);
    //     }
    //     grid[pidx] = 1;
    // }
    
    // free(grid);
    return 0;
}
