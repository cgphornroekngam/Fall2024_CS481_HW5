#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

double gettime(void) {
    struct timeval tval;
    gettimeofday(&tval, NULL);
    return (double)tval.tv_sec + (double)tval.tv_usec / 1000000.0;
}

int getNeighborCount(int *grid, int x, int y, int cols, int rows) {
    int count = 0;
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            if (dx == 0 && dy == 0) continue;  // Skip the current cell
            count += grid[(y + dy) * cols + (x + dx)];
        }
    }
    return count;
}

void nextGeneration(int *current, int *next, int rows, int cols, int *change_flag) {
    *change_flag = 0;
    
    for (int y = 1; y <= rows; y++) {
        for (int x = 1; x <= cols; x++) {
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
    }
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
    size_t size = (rows + 2) * (cols + 2) * sizeof(int);  // Include ghost cells

    // Allocate host memory
    h_current = (int *)malloc(size);
    h_next = (int *)malloc(size);

    // Initialize the grid randomly
    initializeGrid(h_current, rows, cols);
    updateGhostCells(h_current, rows, cols);

    int change_flag, lastI;
    for (int i = 0; i < iterations; i++) {
        change_flag = 0;
        nextGeneration(h_current, h_next, rows, cols, &change_flag);
        
        // Break if no changes occurred
        if (change_flag == 0) {
            lastI = i;
            break;
        }

        // Copy the new generation to the current grid
        for (int j = 0; j < size / sizeof(int); j++) {
            h_current[j] = h_next[j];
        }

        lastI = i;
    }

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
    free(h_current);
    free(h_next);

    printf("Program ran successfully. Final matrix saved to %s.\n", argv[3]);
    return 0;
}
