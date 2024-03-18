#include "Header.cuh"

struct Keyboard {
    uint8_t retBufLen;
    uint8_t bufferSize;
    uint8_t* buffer;
    fd_set readfds;
    struct termios raw;
};

void initKeyboard(struct Keyboard* keyboard, uint8_t bufferSize = 32) {
    keyboard->bufferSize = bufferSize;
    keyboard->buffer = (uint8_t*)malloc(bufferSize);
    tcgetattr(STDIN_FILENO, &keyboard->raw);
    keyboard->raw.c_lflag &= ~(ECHO | ICANON);
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &keyboard->raw);
}

void getKeyboardInput(struct Keyboard* keyboard) {
    FD_ZERO(&keyboard->readfds);
    FD_SET(STDIN_FILENO, &keyboard->readfds);
    if (select(STDIN_FILENO + 1, &keyboard->readfds, NULL, NULL, NULL) > 0) {
        memset(keyboard->buffer, 0, keyboard->bufferSize);
        uint8_t bufferLen = read(STDIN_FILENO, keyboard->buffer, keyboard->bufferSize);
        keyboard->retBufLen = 0;
        for (uint8_t i = 0; i < bufferLen;) {
            if (keyboard->buffer[i] == 27 && i + 2 < bufferLen && keyboard->buffer[i+1] == 91) {
                keyboard->buffer[keyboard->retBufLen++] = keyboard->buffer[i+2] | 0x80;
                i += 3;
            } else {
                keyboard->buffer[keyboard->retBufLen++] = keyboard->buffer[i];
                i++;
            }
        }
    }
}