#include "Header.cuh"

struct Noise {
    uint32_t seed1;
    uint32_t seed2;
};

uint32_t genNoise(Noise* noise) {
    noise->seed2 ^= noise->seed1 >> 17;
    noise->seed2 *= 0xbf324c81;
    noise->seed1 ^= noise->seed2 >> 13;
    noise->seed1 *= 0x9c7493ad;
    return noise->seed1;
}

void initNoise(Noise* noise) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    noise->seed1 = tv.tv_sec;
    noise->seed2 = tv.tv_usec;
    for (uint8_t i = 4; i--;) genNoise(noise);
}