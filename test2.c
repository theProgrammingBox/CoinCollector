#include "Header.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <unistd.h>

double seamlessNoise2D(const struct osn_context *ctx, double x, double y, uint32_t gridSize, float zoomOut) {
    const double coef1 = 6.28318530718 / gridSize;
    const double coef2 = gridSize * zoomOut;
    const double u = x * coef1;
    const double v = y * coef1;
    const double nx = cos(u) * coef2;
    const double ny = sin(u) * coef2;
    const double nz = sin(v) * coef2;
    const double nw = cos(v) * coef2;
    return open_simplex_noise4(ctx, nx, ny, nz, nw);
}

uint32_t rand32(uint32_t* seed) {
    *seed ^= *seed >> 16;
    *seed *= 0xBAC57D37;
    *seed ^= *seed >> 16;
    *seed *= 0x24F66AC9;
    return *seed;
}

float func(float x) {
    return 1.3 * x / (x + 0.3);
}

void octaveNoise2D(uint8_t *grid, uint32_t gridSize, uint8_t octaves, float zoomOut, uint32_t* seed) {
    float sines[gridSize];
    float coef2s[octaves];
    uint8_t amplitude[octaves];
    const float coef1 = 6.28318530718 / gridSize;
    for (int i = gridSize; i--;) sines[i] = sin(i * coef1);
    // for (int i = octaves; i--;) coef2s[i] = gridSize * zoomOut * (1 << i);
    for (int i = octaves; i--;) coef2s[i] = 1 * zoomOut * (1 << i);
    for (int i = octaves; i--;) amplitude[i] = 1 << (octaves - i - 1);
    
    struct osn_context *ctxs[octaves];
    for (int i = octaves; i--;) open_simplex_noise(rand32(seed), &ctxs[i]);
    
    float invMax = 1.0f / ((1 << octaves) - 1);
    float sum;
    uint16_t cosIdx2 = gridSize >> 2;
    for (uint32_t y = gridSize; y--; cosIdx2--) {
        const float x2 = sines[cosIdx2];
        const float y2 = sines[y];
        uint16_t cosIdx1 = gridSize >> 2;
        for (uint32_t x = gridSize; x--; cosIdx1--) {
            const float x1 = sines[cosIdx1];
            const float y1 = sines[x];
            sum = 0;
            for (uint8_t i = octaves; i--;) {
                sum += open_simplex_noise4(ctxs[i], x1 * coef2s[i], y1 * coef2s[i], x2 * coef2s[i], y2 * coef2s[i]) * amplitude[i];
            }
            sum *= invMax;
            grid[y * gridSize + x] = (((sum < 0) ? -func(-sum) : func(sum)) + 1) * 4;
        }
    }
    
    for (int i = octaves; i--;) open_simplex_noise_free(ctxs[i]);
}

int main() {
    uint32_t seed = time(NULL);
    uint8_t *grid = (uint8_t *)malloc(0x100000000);
    octaveNoise2D(grid, 0x10000, 1, 1, &seed);
    return 0;
}