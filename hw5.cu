%%cuda

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <time.h>
#include <cuda_runtime.h>

// #define cols 10  // Grid cols
// #define rows 10 // Grid rows
// #define STEPS 100   // Number of generations

double gettime(void) {
    struct timeval tval;
    gettimeofday(&tval, NULL);
    return (double)tval.tv_sec + (double)tval.tv_usec / 1000000.0;
}

__device__ int getNeighborCount(int *grid, int x, int y, int cols, int rows) {
    int count = 0;
    int dx, dy;

    for (dx = -1; dx <= 1; dx++) {
        for (dy = -1; dy <= 1; dy++) {
            if (dx == 0 && dy == 0) continue;  // Skip the current cell

            int nx = (x + dx + cols) % cols;     // Wrap around edges
            int ny = (y + dy + rows) % rows;  // Wrap around edges

            count += grid[ny * cols + nx];
        }
    }
    return count;
}

__global__ void nextGeneration(int *current, int *next, int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= cols || y >= rows) return;  // Bounds check

    int neighbors = getNeighborCount(current, x, y, rows, cols);
    int idx = y * cols + x;

    // Apply the Game of Life rules
    if (current[idx] == 1 && (neighbors < 2 || neighbors > 3)) {
        next[idx] = 0;  // Cell dies
    } else if (current[idx] == 0 && neighbors == 3) {
        next[idx] = 1;  // Cell becomes alive
    } else {
        next[idx] = current[idx];  // No change
    }
}

void initializeGrid(int *grid, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            grid[(i * cols) + j] = rand() % 2;
        }
          // Random initial state (0 or 1)
    }
}

void saveGridToFile(int *grid, const char *filename, int rows, int cols) {
    FILE *file = fopen(filename, "w");  // Open the file in write mode
    if (file == NULL) {
        fprintf(stderr, "Error opening file for writing.\n");
        return;
    }

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            fprintf(file, "%d ", grid[y * cols + x]);  // Write cell state
        }
        fprintf(file, "\n");  // New line after each row
    }

    fclose(file);  // Close the file
}

int main() {
    srand(100);

    // if (argc != 4) {
    //     printf("Usage: %s <matrix_size> <max_iterations> <output_file>\n", argv[0]);
    //     exit(-1);
    // }

    int rows = 10;
    int cols = rows;
    int iterations  = 100;

    int *h_current, *h_next;           // Host grids
    int *d_current, *d_next;           // Device grids

    size_t size = cols * rows * sizeof(int);
    double starttime = gettime();
    // Allocate host memory
    h_current = (int *)malloc(size);
    h_next = (int *)malloc(size);

    // Initialize the grid randomly
    initializeGrid(h_current, rows, cols);

    // Allocate device memory
    cudaMalloc((void **)&d_current, size);
    cudaMalloc((void **)&d_next, size);

    // Copy the initial grid to the device
    cudaMemcpy(d_current, h_current, size, cudaMemcpyHostToDevice);

    // Define CUDA grid and block dimensions
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,(rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Run the simulation for a number of steps
    for (int step = 0; step < iterations; step++) {
        nextGeneration<<<numBlocks, threadsPerBlock>>>(d_current, d_next, rows, cols);

        // Swap pointers
        int *temp = d_current;
        d_current = d_next;
        d_next = temp;
    }

    // Copy the final grid back to the host
    cudaMemcpy(h_current, d_current, size, cudaMemcpyDeviceToHost);

    // Save the final grid to a file
    saveGridToFile(h_current, "output.txt", cols, rows);

    double endtime = gettime();

    // Free memory
    cudaFree(d_current);
    cudaFree(d_next);
    free(h_current);
    free(h_next);

    printf("Simulation completed. Final grid saved to output.txt.\n");
    return 0;
}
