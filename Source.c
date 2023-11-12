#include "Header.h"
#include <unistd.h>

double layerNoise(const struct osn_context *ctx, uint8_t iterations, double x, double y) {
    double value = 0;
    double layerScale = 1;
    double scale = 0;
    for (int i = iterations; i--;) {
        value += open_simplex_noise2(ctx, x * layerScale, y * layerScale) / layerScale;
        layerScale *= 2;
        scale += 1 / layerScale;
    }
    return value / scale;
}

double seamlessNoise4D(const struct osn_context *ctx, double x, double y, double gridWidth, double gridHeight, float frequency) {
    const double TWO_PI = 6.28318530718;
    
    // Convert 2D coordinates to angles for toroidal mapping
    double u = (x / gridWidth) * TWO_PI;
    double v = (y / gridHeight) * TWO_PI;

    // Map onto a torus
    double nx = cos(u) + 0.5 * cos(u) * cos(v) * frequency;
    double ny = sin(u) + 0.5 * sin(u) * cos(v) * frequency;
    double nz = 0.5 * sin(v) * frequency;
    double nw = 0.5 * cos(v) * frequency;

    // Get noise value from 4D noise function
    return open_simplex_noise4(ctx, nx, ny, nz, nw);
}

double octaveNoise4D(const struct osn_context *ctx, double x, double y, double gridWidth, double gridHeight, int octaves) {
    double value = 0;
    double frequency = 1.0;
    double amplitude = 1.0;
    double max = 0; // Used for normalizing result to 0.0 - 1.0

    for (int i = 0; i < octaves; i++) {
        value += seamlessNoise4D(ctx, x * frequency, y * frequency, gridWidth, gridHeight, 1) * amplitude;
        max += amplitude;

        amplitude *= 0.5;
        frequency *= 2.0;
    }

    return value / max; // Normalize the result
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
        // grid[i] = (octaveNoise4D(ctx, x, y, 0x100, 0x100, 1) + 1) * 4;
        grid[i] = (open_simplex_noise2(ctx, x * 0.2, y * 0.2) + 1) * 4;
    }
    grid[0] = 1;
    
    while(1) {
        system("clear");
        for (uint8_t i = VIEW_SIZE, ry = y + VIEW_RADIUS; i--; ry--) {
            for (uint8_t j = VIEW_SIZE, rx = x + VIEW_RADIUS; j--; rx--) {
                switch (grid[ry << 8 | rx]) {
                case 0: printf("  "); break;
                case 1: printf(".."); break;
                case 2: printf(",,"); break;
                case 3: printf("::"); break;
                case 4: printf("ii"); break;
                case 5: printf("ll"); break;
                case 6: printf("ww"); break;
                case 7: printf("WW"); break;
                }
            }
            printf("\n");
        }
        
        // printf("Move (wasd): ");
        // scanf(" %c", &move);
        
        x++;
        y++;
        // x += (move == 'a') - (move == 'd');
        // y += (move == 'w') - (move == 's');
        pidx = y << 8 | x;
        usleep(80000);
    }
    
    free(grid);
    return 0;
}
