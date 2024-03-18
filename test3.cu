#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <termios.h>

int main() {
    const int BUFFER_SIZE = 256;
    char buffer[BUFFER_SIZE];
    int buffer_len = 0;
    
    struct termios raw;
    tcgetattr(STDIN_FILENO, &raw);
    raw.c_lflag &= ~(ECHO | ICANON);
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);
    
    fd_set readfds;
    // struct timeval timeout = {0, 0};

    while (1) {
        printf("Start\n");
        memset(buffer, 0, BUFFER_SIZE);
        buffer_len = 0;

        // sleep(2); // Simulate doing something else for 2 second.
        
        FD_ZERO(&readfds);
        FD_SET(STDIN_FILENO, &readfds);
        if (select(STDIN_FILENO + 1, &readfds, NULL, NULL, NULL) > 0) {
            buffer_len = read(STDIN_FILENO, buffer, BUFFER_SIZE);
            for (int i = 0; i < buffer_len;) {
                if (buffer[i] == 27 && i + 2 < buffer_len && buffer[i+1] == 91) {
                    printf("-%d\n", buffer[i+2]);
                    i += 3;
                } else {
                    printf("%d\n", buffer[i]);
                    i++;
                }
            }
        }
    }

    return 0;
}