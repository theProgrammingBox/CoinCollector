#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define GRID_SIZE 9
#define NUM_COINS 9
#define VIEW_RADIUS 4

unsigned int customRand(unsigned int* seed) {
    *seed *= 0xBAC57D37;
    *seed ^= *seed >> 16;
    *seed *= 0x24F66AC9;
    *seed ^= *seed >> 16;
    return *seed;
}

void placeCoin(char *grid, unsigned int* seed) {
    int pos;
    do {
        pos = customRand(seed) % (GRID_SIZE * GRID_SIZE);
    } while (grid[pos] != '_');
    grid[pos] = 'C';
}

int main() {
    unsigned int seed = time(NULL) ^ 0xE621B963;
    char *grid = malloc(GRID_SIZE * GRID_SIZE * sizeof(char));
    int x = 0, y = 0, coinsCollected = 0;
    char move;

    for (int i = 0; i < 8; i++) customRand(&seed);
    memset(grid, '_', GRID_SIZE * GRID_SIZE * sizeof(char));
    grid[0] = 'P';

    for (int i = 0; i < NUM_COINS; ++i) placeCoin(grid, &seed);

    while (1) {
        system("clear");
        for (int i = -VIEW_RADIUS; i <= VIEW_RADIUS; i++) {
            for (int j = -VIEW_RADIUS; j <= VIEW_RADIUS; j++) {
                int row = (x + i + GRID_SIZE) % GRID_SIZE;
                int col = (y + j + GRID_SIZE) % GRID_SIZE;
                printf("%c ", grid[row * GRID_SIZE + col]);
            }
            printf("\n");
        }
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
