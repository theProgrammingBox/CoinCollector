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

double octaveNoise2D(const struct osn_context *ctx, double x, double y, double gridWidth, double gridHeight, float frequency, int octaves, float persistence) {
    double total = 0;
    double frequencyAccumulator = frequency;
    double amplitudeAccumulator = 1;
    double maxValue = 0;

    for (int i = 0; i < octaves; i++) {
        total += seamlessNoise2D(ctx, x, y, gridWidth, gridHeight, frequencyAccumulator) * amplitudeAccumulator;
        maxValue += amplitudeAccumulator;
        amplitudeAccumulator *= persistence;
        frequencyAccumulator *= 2;
    }

    return total / maxValue;
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

    uint32_t seed = time(NULL) ^ 0x5EED5EED;
    for (int i = 16; i--;) rand16(&seed);
    struct osn_context *ctx;
    open_simplex_noise(seed, &ctx);
    
    uint8_t *grid = (uint8_t *)malloc(0x10000);
    uint8_t x = 0, y = 0, move;
    uint16_t pidx = 0;
    
    for (uint32_t i = 0x10000; i--;) {
        double x = (i & 0xFF);
        double y = (i >> 8);
        // grid[i] = (seamlessNoise2D(ctx, x, y, 0x100, 0x100, 4) + 1) * 4;
        grid[i] = (octaveNoise2D(ctx, x, y, 0x100, 0x100, 4, 4, 0.5) + 1) * 4;
    }
    
    while(1) {
        system("clear");
        for (uint8_t i = VIEW_SIZE, ry = y + VIEW_RADIUS; i--; ry--) {
            for (uint8_t j = VIEW_SIZE, rx = x + VIEW_RADIUS; j--; rx--) {
                switch (grid[ry << 8 | rx]) {
                    case 0: printf("\x1b[38;2;000;000;139m..\x1b[0m"); break;
                    case 1: printf("\x1b[38;2;000;105;148m--\x1b[0m"); break;
                    case 2: printf("\x1b[38;2;173;216;230m;;\x1b[0m"); break;
                    case 3: printf("\x1b[38;2;194;178;128m==\x1b[0m"); break;
                    case 4: printf("\x1b[38;2;155;118;083m**\x1b[0m"); break;
                    case 5: printf("\x1b[38;2;100;200;100m++\x1b[0m"); break;
                    case 6: printf("\x1b[38;2;010;120;010m##\x1b[0m"); break;
                    case 7: printf("\x1b[38;2;000;080;000m@@\x1b[0m"); break;
                }
                // switch (grid[ry << 8 | rx] | ((i & 1) << 3)) {
                //     case 0: printf("\x1b[38;2;000;000;139m░ \x1b[0m"); break;
                //     case 1: printf("\x1b[38;2;000;105;148m░░\x1b[0m"); break;
                //     case 2: printf("\x1b[38;2;173;216;230m▒░\x1b[0m"); break;
                //     case 3: printf("\x1b[38;2;194;178;128m▒▒\x1b[0m"); break;
                //     case 4: printf("\x1b[38;2;155;118;083m▓▒\x1b[0m"); break;
                //     case 5: printf("\x1b[38;2;100;200;100m▓▓\x1b[0m"); break;
                //     case 6: printf("\x1b[38;2;010;120;010m█▓\x1b[0m"); break;
                //     case 7: printf("\x1b[38;2;000;080;000m██\x1b[0m"); break;
                    
                //     case 8: printf("\x1b[38;2;000;000;139m ░\x1b[0m"); break;
                //     case 9: printf("\x1b[38;2;000;105;148m░░\x1b[0m"); break;
                //     case 10: printf("\x1b[38;2;173;216;230m░▒\x1b[0m"); break;
                //     case 11: printf("\x1b[38;2;194;178;128m▒▒\x1b[0m"); break;
                //     case 12: printf("\x1b[38;2;155;118;083m▒▓\x1b[0m"); break;
                //     case 13: printf("\x1b[38;2;100;200;100m▓▓\x1b[0m"); break;
                //     case 14: printf("\x1b[38;2;010;120;010m▓█\x1b[0m"); break;
                //     case 15: printf("\x1b[38;2;000;080;000m██\x1b[0m"); break;
                // }
            }
            printf("\n");
        }

        printf("Move (wasd): ");
        scanf(" %c", &move);
        
        // x++;
        // y++;
        // usleep(80000);

        x += ((move == 'a') - (move == 'd')) * 16;
        y += ((move == 'w') - (move == 's')) * 16;
        pidx = y << 8 | x;
    }
    
    free(grid);
    return 0;
}
