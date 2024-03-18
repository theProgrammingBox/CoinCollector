#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <termios.h>
#include <sys/select.h>
#include <fcntl.h>
#include <stdint.h>

#define BUFFER_SIZE 256 // Adjust for expected input burst size

void make_non_blocking(int fd) {
    int flags = fcntl(fd, F_GETFL, 0);
    fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}

int main() {
    char buffer[BUFFER_SIZE];
    int buffer_len = 0;
    
    struct termios raw;
    tcgetattr(STDIN_FILENO, &raw);
    raw.c_lflag &= ~(ECHO | ICANON);
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);

    while (1) {
        printf("Start\n");
        memset(buffer, 0, BUFFER_SIZE);
        buffer_len = 0;

        // Perform other stuff for 1 second
        sleep(1);
        
        fd_set readfds;
        FD_ZERO(&readfds);
        FD_SET(STDIN_FILENO, &readfds);
        struct timeval timeout = {0, 0};
        uint8_t notDone = select(STDIN_FILENO + 1, &readfds, NULL, NULL, &timeout) > 0;
         while (notDone) {
            buffer_len += read(STDIN_FILENO, buffer + buffer_len, BUFFER_SIZE - buffer_len);
            notDone = select(STDIN_FILENO + 1, &readfds, NULL, NULL, &timeout) > 0;
        }

        int i = 0;
        while (i < buffer_len) {
            if (buffer[i] == 27 && (i + 2 < buffer_len) && buffer[i+1] == '[') { // Start of an escape sequence
                // Print the sequence as a single group
                printf("Read");
                for (int j = 0; j < 3 && (i + j) < buffer_len; j++) {
                    printf(" %d", buffer[i + j]);
                }
                printf("\n");
                i += 3;
            } else {
                printf("Read %d\n", buffer[i]);
                i++;
            }
        }
    }

    return 0;
}
