#include "Header.h"
#include <unistd.h>

// add width to frequency
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

uint32_t rand32(uint32_t* seed) {
    *seed ^= *seed >> 16;
    *seed *= 0xBAC57D37;
    *seed ^= *seed >> 16;
    *seed *= 0x24F66AC9;
    return *seed;
}

double func(double x) {
    return 1.3 * x / (x + 0.3);
}

double octaveNoise2D(uint8_t *grid, int gridWidth, int gridHeight, int octaves, float zoomOut) {
    uint32_t seed1 = time(NULL);
    
    struct osn_context **ctx1;
    ctx1 = (struct osn_context **)malloc(octaves * sizeof(struct osn_context *));
    for (int i = octaves; i--;) {
        open_simplex_noise(rand32(&seed1), &ctx1[i]);
    }
    
    double max = 0;
    double frequency = zoomOut;
    for (int i = octaves; i--;) {
        max += 1 / frequency;
        frequency *= 2;
    }
    
    double sum;
    for (int y = gridHeight; y--;) {
        for (int x = gridWidth; x--;) {
            sum = 0;
            frequency = zoomOut;
            for (int i = octaves; i--;) {
                sum += seamlessNoise2D(ctx1[i], x, y, gridWidth, gridHeight, frequency) / frequency;
                frequency *= 2;
            }
            sum = sum / max;
            if (sum < 0) sum = func(-sum) * -1;
            else sum = func(sum);
            grid[y * gridWidth + x] = (sum + 1) * 4;
        }
    }
    
    for (int i = octaves; i--;) {
        open_simplex_noise_free(ctx1[i]);
    }
    free(ctx1);
}

int main() {
    const uint8_t VIEW_RADIUS = 16;
    const uint8_t VIEW_SIZE = VIEW_RADIUS * 2 + 1;

    uint8_t *grid = (uint8_t *)malloc(0x10000);
    uint8_t x = 0, y = 0, move;
    uint16_t pidx = 0;
    
    octaveNoise2D(grid, 0x100, 0x100, 4, 8);
    
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
