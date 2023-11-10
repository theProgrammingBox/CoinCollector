#include <stdio.h>

#define GRID_SIZE 8
#define NUM_COINS 8

unsigned int randomSeed(unsigned int* seed) {
    *seed *= 0xBAC57D37;
    *seed ^= *seed >> 16;
    *seed *= 0x24F66AC9;
    *seed ^= *seed >> 16;
    return *seed;
}

void placeCoin(char *grid, unsigned int* seed) {
    int pos;
    do {
        pos = randomSeed(seed) % (GRID_SIZE * GRID_SIZE);
    } while (grid[pos] != '_');
    grid[pos] = 'C';
}

int main() {
    unsigned int seed = time(NULL) ^ 0xE621B963;
    char *grid = malloc(GRID_SIZE * GRID_SIZE * sizeof(char));
    int x = 0, y = 0, coinsCollected = 0;
    char move;

    for (int i = 0; i < 8; i++) randomSeed(&seed);
    memset(grid, '_', GRID_SIZE * GRID_SIZE * sizeof(char));
    grid[0] = 'P';

    for (int i = 0; i < NUM_COINS; ++i) placeCoin(grid, &seed);

    while (1) {
        system("clear");
        for (int i = 0; i < GRID_SIZE; i++, printf("\n"))
            for (int j = 0; j < GRID_SIZE; j++)
                printf("%c ", grid[i * GRID_SIZE + j]);
        printf("\nCoins Collected: %d\n", coinsCollected);
        printf("Move (WASD): ");
        scanf(" %c", &move);
        
        int newX = (x + (move == 's') - (move == 'w') + GRID_SIZE) % GRID_SIZE;
        int newY = (y + (move == 'd') - (move == 'a') + GRID_SIZE) % GRID_SIZE;
        int newIndex = newX * GRID_SIZE + newY;

        if (grid[newIndex] == 'C') {
            coinsCollected++;
            placeCoin(grid, &seed);
        }

        grid[x * GRID_SIZE + y] = '_';
        grid[newIndex] = 'P';
        x = newX;
        y = newY;
    }

    free(grid);
    return 0;
}
