#include "Header.cuh"

struct Noise {
    uint32_t seed1;
    uint32_t seed2;
};

void mix(Noise* noise) {
    noise->seed2 ^= noise->seed1 >> 17;
    noise->seed2 *= 0xbf324c81;
    noise->seed1 ^= noise->seed2 >> 13;
    noise->seed1 *= 0x9c7493ad;
}

uint32_t genUint(Noise* noise) {
    mix(noise);
    return noise->seed1;
}

float genUniform(Noise* noise) {
    mix(noise);
    return (float)noise->seed1 * 0.00000000023283064359965952028f;
}

float genNormal(Noise* noise) {
    mix(noise);
    return sqrtf(-2 * logf((float)noise->seed1 / 0xffffffff)) * cosf(6.283185307179586476925286766559f / 0xffffffff * noise->seed2);
}

void initNoise(Noise* noise) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    noise->seed1 = tv.tv_sec;
    noise->seed2 = tv.tv_usec;
    for (uint8_t i = 4; i--;) mix(noise);
}