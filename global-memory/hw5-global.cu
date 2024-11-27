#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 16  // CUDA Kernel block size

double gettime(void) {
    struct timeval tval;
    gettimeofday(&tval, NULL);
    return (double)tval.tv_sec + (double)tval.tv_usec / 1000000.0;
}

__device__ int getNeighborCount(int *grid, int x, int y, int cols, int rows) {
    int count = 0;
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            if (dx == 0 && dy == 0) continue;  // Skip the current cell
            count += grid[(y + dy) * cols + (x + dx)];
        }
    }
    return count;
}

__global__ void nextGeneration(int *current, int *next, int rows, int cols, int *change_flag) {
    int x = blockIdx.x * blockDim.x + threadIdx.x + 1;  // Offset by 1 for ghost cells
    int y = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (x > cols || y > rows) return;  // Bounds check (excluding ghost cells)

    int neighbors = getNeighborCount(current, x, y, cols + 2, rows + 2);  // Adjust grid size for ghost cells
    int idx = y * (cols + 2) + x;

    // Apply the Game of Life rules
    int new_state = current[idx];
    if (current[idx] == 1 && (neighbors < 2 || neighbors > 3)) {
        new_state = 0;  // Cell dies
    } else if (current[idx] == 0 && neighbors == 3) {
        new_state = 1;  // Cell becomes alive
    }

    if (new_state != current[idx]) {
        *change_flag = 1;  // Mark that a change has occurred
    }
    
    next[idx] = new_state;
}

void initializeGrid(int *grid, int rows, int cols) {
    for (int i = 1; i <= rows; i++) {
        for (int j = 1; j <= cols; j++) {
            grid[i * (cols + 2) + j] = rand() % 2;  // Initialize inner grid randomly
        }
    }
}

void updateGhostCells(int *grid, int rows, int cols) {
    // Copy edges to ghost cells
    for (int i = 1; i <= rows; i++) {
        grid[i * (cols + 2)] = grid[i * (cols + 2) + cols];           // Left ghost column
        grid[i * (cols + 2) + cols + 1] = grid[i * (cols + 2) + 1];   // Right ghost column
    }
    for (int j = 1; j <= cols; j++) {
        grid[j] = grid[rows * (cols + 2) + j];                        // Top ghost row
        grid[(rows + 1) * (cols + 2) + j] = grid[(1) * (cols + 2) + j]; // Bottom ghost row
    }

    // Corners
    grid[0] = grid[rows * (cols + 2)];                                // Top-left corner
    grid[(cols + 1)] = grid[rows * (cols + 2) + cols];                // Top-right corner
    grid[(rows + 1) * (cols + 2)] = grid[(1) * (cols + 2)];           // Bottom-left corner
    grid[(rows + 1) * (cols + 2) + (cols + 1)] = grid[(1) * (cols + 2) + cols];  // Bottom-right corner
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
    int iterations = atoi(argv[2]);

    int *h_current, *h_next;
    int *d_current, *d_next;
    int *d_change_flag, h_change_flag;
    size_t size = (rows + 2) * (cols + 2) * sizeof(int);  // Include ghost cells

    // Allocate host memory
    h_current = (int *)malloc(size);
    h_next = (int *)malloc(size);

    // Initialize the grid randomly
    initializeGrid(h_current, rows, cols);
    updateGhostCells(h_current, rows, cols);

    // Allocate device memory
    cudaMalloc((void **)&d_current, size);
    cudaMalloc((void **)&d_next, size);
    cudaMalloc((void **)&d_change_flag, sizeof(int));

    // Copy the initial grid to the device
    cudaMemcpy(d_current, h_current, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 numBlocks((cols + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, (rows + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    
    int lastI;
    for (int i = 0; i < iterations; i++) {
        h_change_flag = 0;
        cudaMemcpy(d_change_flag, &h_change_flag, sizeof(int), cudaMemcpyHostToDevice);
        
        nextGeneration<<<numBlocks, threadsPerBlock>>>(d_current, d_next, rows, cols, d_change_flag);
        cudaMemcpy(&h_change_flag, d_change_flag, sizeof(int), cudaMemcpyDeviceToHost);
        
        // Break if no changes occurred
        if (h_change_flag == 0) {
            lastI = i;
            break;
        }
        
        cudaMemcpy(d_current, d_next, size, cudaMemcpyDeviceToDevice);
        lastI = i;
    }

    // Copy the final grid back to the host
    cudaMemcpy(h_current, d_current, size, cudaMemcpyDeviceToHost);

    double endtime = gettime();

    // Save final board to file
    FILE *outFile = fopen(argv[3], "w");
    if (outFile == NULL) {
        fprintf(stderr, "Error opening file for writing.\n");
        return -1;
    }
    fprintf(outFile, "======\nGEN %d\n======\n", lastI + 1);
    for (int y = 1; y <= rows; y++) {
        for (int x = 1; x <= cols; x++) {
            fprintf(outFile, "%d ", h_current[y * (cols + 2) + x]);
        }
        fprintf(outFile, "\n");
    }
    fprintf(outFile, "\nTime taken = %lf seconds\n", endtime - starttime);
    fclose(outFile);

    // free memory
    cudaFree(d_current);
    cudaFree(d_next);
    cudaFree(d_change_flag);
    free(h_current);
    free(h_next);

    printf("Program ran successfully. Final matrix saved to %s.\n", argv[3]);
    return 0;
}
