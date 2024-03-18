#include "Keyboard.cuh"

int main() {
    struct Keyboard keyboard;
    initKeyboard(&keyboard);

    while (1) {
        getKeyboardInput(&keyboard);
        for (uint8_t i = 0; i < keyboard.retBufLen; i++) {
            printf("%d\n", keyboard.buffer[i]);
        }
    }

    return 0;
}
