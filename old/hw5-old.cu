#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <time.h>
#include <cuda.h>

// Prints the life matrix to a file
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

// Initialize all arrays to 0
void initArr(int r, int c, char *arr) {
    for (int i = 0; i < r + 2; i++) {
        for (int j = 0; j < c + 2; j++) {
            arr[i * (c + 2) + j] = '0';
        }
    }
}

// Generate a random starting life matrix
void fillLife(int r, int c, char *arr) {
    for (int i = 1; i < r + 1; i++) {
        for (int j = 1; j < c + 1; j++) {
            int temp = rand() % 2;
            arr[i * (c + 2) + j] = temp ? '1' : '0';
        }
    }
}

// Kernel for simulating life according to Game of Life rules
__global__ void simLifeKernel(int r, int c, char *oldArr, char *newArr, int *changed) {
    int iR = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int jC = blockIdx.x * blockDim.x + threadIdx.x + 1;

    if (iR <= r && jC <= c) {
        int aliveNeighbors = 0;
        int index = iR * (c + 2) + jC;

        // Count alive neighbors
        // Top row
        aliveNeighbors += (oldArr[(iR - 1) * (c + 2) + (jC - 1)] == '1');
        aliveNeighbors += (oldArr[(iR - 1) * (c + 2) + jC] == '1');
        aliveNeighbors += (oldArr[(iR - 1) * (c + 2) + (jC + 1)] == '1');

        // Mid row
        aliveNeighbors += (oldArr[iR * (c + 2) + (jC - 1)] == '1');
        aliveNeighbors += (oldArr[iR * (c + 2) + (jC + 1)] == '1');

        // Bottom row
        aliveNeighbors += (oldArr[(iR + 1) * (c + 2) + (jC - 1)] == '1');
        aliveNeighbors += (oldArr[(iR + 1) * (c + 2) + jC] == '1');
        aliveNeighbors += (oldArr[(iR + 1) * (c + 2) + (jC + 1)] == '1');

        // Game of Life rules
        if (oldArr[index] == '1') {
            if (aliveNeighbors <= 1 || aliveNeighbors >= 4) {
                newArr[index] = '0';
                *changed = 1;
            }
        } else if (oldArr[index] == '0') {
            if (aliveNeighbors == 3) {
                newArr[index] = '1';
                *changed = 1;
            }
        } else {
            newArr[index] = oldArr[index];
        }
    }
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

    // Allocate host matrices
    char *oldMatrix = (char*)malloc((rows + 2) * (cols + 2) * sizeof(char));
    char *lifeMatrix = (char*)malloc((rows + 2) * (cols + 2) * sizeof(char));
    int *changed = (int*)malloc(sizeof(int));

    initArr(rows, cols, oldMatrix);
    fillLife(rows, cols, oldMatrix);
    // printf("gen 0:\n");
    // printMatrix(rows, cols, oldMatrix);

    // Allocate device matrices
    char *d_oldMatrix, *d_lifeMatrix;
    int *d_changed;
    cudaMalloc(&d_oldMatrix, (rows + 2) * (cols + 2) * sizeof(char));
    cudaMalloc(&d_lifeMatrix, (rows + 2) * (cols + 2) * sizeof(char));
    cudaMalloc(&d_changed, sizeof(int));

    cudaMemcpy(d_oldMatrix, oldMatrix, (rows + 2) * (cols + 2) * sizeof(char), cudaMemcpyHostToDevice);

    int lastI = 0;
    dim3 blockDim(16, 16);
    dim3 gridDim((cols + 15) / 16, (rows + 15) / 16);

    for (int i = 0; i < atoi(argv[2]); i++) {
        *changed = 0;
        cudaMemcpy(d_changed, changed, sizeof(int), cudaMemcpyHostToDevice);

        simLifeKernel<<<gridDim, blockDim>>>(rows, cols, d_oldMatrix, d_lifeMatrix, d_changed);
        cudaDeviceSynchronize();

        cudaMemcpy(changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost);
        if (!(*changed)) {
            lastI = i;
            break;
        }

        // Swap matrices
        char *temp = d_oldMatrix;
        d_oldMatrix = d_lifeMatrix;
        d_lifeMatrix = temp;
    }

    cudaMemcpy(lifeMatrix, d_oldMatrix, (rows + 2) * (cols + 2) * sizeof(char), cudaMemcpyDeviceToHost);
    double endtime = gettime();

    FILE* outFile = fopen(argv[3], "w");
    fprintf(outFile, "\n======\nGEN %d\n======\n", lastI + 1);
    fprintfLife(rows, cols, lifeMatrix, outFile);
    fprintf(outFile, "Time taken = %lf seconds\n", endtime - starttime);

    free(oldMatrix);
    free(lifeMatrix);
    free(changed);
    cudaFree(d_oldMatrix);
    cudaFree(d_lifeMatrix);
    cudaFree(d_changed);
    fclose(outFile);

    return 0;
}
