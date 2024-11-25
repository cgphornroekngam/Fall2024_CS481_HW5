#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <time.h>

// prints only the life matrix
void fprintfLife(int r, int c, char *arr, FILE* file) {
    for (int i = 1; i < r + 1; i++) {
        for (int j = 1; j < c + 1; j++) {
            fprintf(file, "%c ", arr[i * (c + 2) + j]);
        }
        fprintf(file, "\n");
    }
}

void printMatrix(int rows, int cols, char *arr) {
    for (int i = 1; i < rows + 1; i++) {
        for (int j = 1; j < cols + 1; j++) {
            printf("%c ", arr[i * (cols + 2) + j]);
        }
        printf("\n");
    }
}

double gettime(void) {
    struct timeval tval;
    gettimeofday(&tval, NULL);
    return (double)tval.tv_sec + (double)tval.tv_usec / 1000000.0;
}

// inits all arrays to 0
void initArr(int r, int c, char *arr) {
    for (int i = 0; i < r + 2; i++) {
        for (int j = 0; j < c + 2; j++) {
            arr[i * (c + 2) + j] = '0';
        }
    }
}

// generates a random starting life matrix
void fillLife(int r, int c, char *arr) {
    for (int i = 1; i < r + 1; i++) {
        for (int j = 1; j < c + 1; j++) {
            int temp = rand() % 2;
            arr[i * (c + 2) + j] = temp ? '1' : '0';
        }
    }
}

// copies matrix from old to new arr
void copyMatrix(int r, int c, char *oldArr, char *newArr) {
    for (int i = 1; i < r + 1; i++) {
        for (int j = 1; j < c + 1; j++) {
            newArr[i * (c + 2) + j] = oldArr[i * (c + 2) + j];
        }
    }
}

// sims life according to game rules and returns if changed
int simLife(int r, int c, char *oldArr, char *newArr) {
    int changed = 0;
    for (int iR = 1; iR < r + 1; iR++) {
        for (int jC = 1; jC < c + 1; jC++) {
            int aliveNeighbors = 0;
            int index = iR * (c + 2) + jC;

            // counting alive neighbors
            // top row
            if (oldArr[(iR - 1) * (c + 2) + (jC - 1)] == '1') aliveNeighbors++;
            if (oldArr[(iR - 1) * (c + 2) + jC] == '1') aliveNeighbors++;
            if (oldArr[(iR - 1) * (c + 2) + (jC + 1)] == '1') aliveNeighbors++;

            // mid row
            if (oldArr[iR * (c + 2) + (jC - 1)] == '1') aliveNeighbors++;
            if (oldArr[iR * (c + 2) + (jC + 1)] == '1') aliveNeighbors++;

            // bot row
            if (oldArr[(iR + 1) * (c + 2) + (jC - 1)] == '1') aliveNeighbors++;
            if (oldArr[(iR + 1) * (c + 2) + jC] == '1') aliveNeighbors++;
            if (oldArr[(iR + 1) * (c + 2) + (jC + 1)] == '1') aliveNeighbors++;

            // alive case
            if (oldArr[index] == '1') {
                if (aliveNeighbors <= 1 || aliveNeighbors >= 4) {
                    newArr[index] = '0';
                    changed = 1;
                }
            }
            // dead case
            else if (oldArr[index] == '0') {
                if (aliveNeighbors == 3) {
                    newArr[index] = '1';
                    changed = 1;
                }
            }
        }
    }
    return changed;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        printf("Usage: %s <matrix_size> <max_iterations> <output_file>\n", argv[0]);
        exit(-1);
    }

    double starttime = gettime();
    srand(100);

    int rows = atoi(argv[1]);
    int cols = rows;

    // Allocate matrices as 1D arrays
    char *lifeMatrix = (char*)malloc((rows + 2) * (cols + 2) * sizeof(char));
    char *oldMatrix = (char*)malloc((rows + 2) * (cols + 2) * sizeof(char));

    initArr(rows, cols, oldMatrix);
    fillLife(rows, cols, oldMatrix);
    printf("gen 0:\n");
    printMatrix(rows, cols, oldMatrix);
    copyMatrix(rows, cols, oldMatrix, lifeMatrix);

    int lastI = 0;
    int hasChanged = 1;
    for (int i = 0; i < atoi(argv[2]); i++) {
        hasChanged = simLife(rows, cols, oldMatrix, lifeMatrix);
        lastI = i;

        if (!hasChanged) {
            break;
        }

        copyMatrix(rows, cols, lifeMatrix, oldMatrix);
    }

    double endtime = gettime();

    FILE* outFile = fopen(argv[3], "w");
    fprintf(outFile, "\n======\nGEN %d\n======\n", lastI + 1);
    fprintfLife(rows, cols, lifeMatrix, outFile);
    if (hasChanged == 0) {
        fprintf(outFile, "No changes b/w GEN %d and GEN %d\nLast viable GEN is GEN %d\n", lastI, lastI + 1, lastI);
    }
    fprintf(outFile, "Time taken = %lf seconds\n", endtime - starttime);

    free(lifeMatrix);
    free(oldMatrix);
    fclose(outFile);

    return 0;
}
