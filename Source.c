#include "Header.h"
#include <unistd.h>

double seamlessNoise2D(const struct osn_context *ctx, double x, double y, double gridWidth, double gridHeight, float frequency) {
    const double TWO_PI = 6.28318530718;
    
    double u = (x / gridWidth) * TWO_PI;
    double v = (y / gridHeight) * TWO_PI;

    double nx = cos(u) * frequency;
    double ny = sin(u) * frequency;
    double nz = sin(v) * frequency;
    double nw = cos(v) * frequency;

    return open_simplex_noise4(ctx, nx, ny, nz, nw);
}

uint16_t rand16(uint32_t* seed) {
    *seed ^= *seed >> 16;
    *seed *= 0xBAC57D37;
    *seed ^= *seed >> 16;
    *seed *= 0x24F66AC9;
    return *seed;
}

int main() {
    const uint8_t VIEW_RADIUS = 16;
    const uint8_t VIEW_SIZE = VIEW_RADIUS * 2 + 1;

    uint32_t seed = time(NULL);
    for (int i = 16; i--;) rand16(&seed);
    struct osn_context *ctx;
    open_simplex_noise(seed, &ctx);
    
    uint8_t *grid = (uint8_t *)malloc(0x10000);
    uint8_t x = 0, y = 0, move;
    uint16_t pidx = 0;
    
    for (uint32_t i = 0x10000; i--;) {
        double x = (i & 0xFF);
        double y = (i >> 8);
        grid[i] = (seamlessNoise1D(ctx, x, y, 0x100, 0x100, 8) + 1) * 4;
    }
    grid[0] = 1;
    
    // while(1) {
        system("clear");
        for (uint8_t i = VIEW_SIZE, ry = y + VIEW_RADIUS; i--; ry--) {
            for (uint8_t j = VIEW_SIZE, rx = x + VIEW_RADIUS; j--; rx--) {
                switch (grid[ry << 8 | rx]) {
                
                case 0: printf("\x1b[38;2;000;000;128m..\x1b[0m"); break; // Deep Water (Dark Blue)
                case 1: printf("\x1b[38;2;135;206;235m--\x1b[0m"); break; // Shallow Water (Sky Blue)
                case 2: printf("\x1b[38;2;210;180;140m;;\x1b[0m"); break; // Sand (Tan)
                case 3: printf("\x1b[38;2;000;140;000m==\x1b[0m"); break; // Grass (Green)
                case 4: printf("\x1b[38;2;000;080;000m++\x1b[0m"); break; // Forest (Darker Green)
                case 5: printf("\x1b[38;2;139;069;019m**\x1b[0m"); break; // Dirt or Earth (Brown)
                case 6: printf("\x1b[38;2;128;128;128m##\x1b[0m"); break; // Rocky Terrain (Gray)
                case 7: printf("\x1b[38;2;255;255;255m@@\x1b[0m"); break; // Snow or High Altitude (White)
                }
            }
            printf("\n");
        }
        
        // printf("Move (wasd): ");
        // scanf(" %c", &move);
        
        // x++;
        // y++;
        // usleep(80000);

        // x += (move == 'a') - (move == 'd');
        // y += (move == 'w') - (move == 's');
        pidx = y << 8 | x;
    // }
    
    free(grid);
    return 0;
}
