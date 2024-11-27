/*
Name: Chris Phornroekngam
Email: cgphornroekngam@crimson.ua.edu
Course Section: CS 481
Homework #: 4
To Compile: mpicc -o hw4 hw4.c
To Run: mpirun -np <#proc> ./hw4 <size> <iterations> <output_file>
*/
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <time.h>
#include <mpi.h>

void initArr(int rows, int cols, char *arr) {
    for (int i = 0; i < (rows + 2) * (cols + 2); i++) {
        arr[i] = '0';
    }
}

void fillLife(int rows, int cols, char *arr) {
    for (int i = 1; i < rows + 1; i++) {
        for (int j = 1; j < cols + 1; j++) {
            arr[i * (cols + 2) + j] = (rand() % 2) ? '1' : '0';
        }
    }
}

int simLife(int rows, int cols, char *oldArr, char *newArr, int startRow, int endRow) {
    int changed = 0;
    for (int iR = startRow; iR <= endRow; iR++) {
        for (int jC = 1; jC < cols + 1; jC++) {
            int aliveNeighbors = 0;
            int index = iR * (cols + 2) + jC;

            // Count alive neighbors
            if (oldArr[(iR - 1) * (cols + 2) + (jC - 1)] == '1') aliveNeighbors++;
            if (oldArr[(iR - 1) * (cols + 2) + jC] == '1') aliveNeighbors++;
            if (oldArr[(iR - 1) * (cols + 2) + (jC + 1)] == '1') aliveNeighbors++;
            if (oldArr[iR * (cols + 2) + (jC - 1)] == '1') aliveNeighbors++;
            if (oldArr[iR * (cols + 2) + (jC + 1)] == '1') aliveNeighbors++;
            if (oldArr[(iR + 1) * (cols + 2) + (jC - 1)] == '1') aliveNeighbors++;
            if (oldArr[(iR + 1) * (cols + 2) + jC] == '1') aliveNeighbors++;
            if (oldArr[(iR + 1) * (cols + 2) + (jC + 1)] == '1') aliveNeighbors++;

            if (oldArr[index] == '1') {
                newArr[index] = (aliveNeighbors <= 1 || aliveNeighbors >= 4) ? '0' : '1';
                changed |= (newArr[index] != oldArr[index]);
            } else {
                newArr[index] = (aliveNeighbors == 3) ? '1' : '0';
                changed |= (newArr[index] != oldArr[index]);
            }
        }
    }
    return changed;
}

void fprintfLife(int rows, int cols, char *arr, FILE *file) {
    for (int i = 1; i < rows + 1; i++) {
        for (int j = 1; j < cols + 1; j++) {
            fprintf(file, "%c ", arr[i * (cols + 2) + j]);
        }
        fprintf(file, "\n");
    }
}

double gettime(void) {
    struct timeval tval;
    gettimeofday(&tval, NULL);
    return (double)tval.tv_sec + (double)tval.tv_usec / 1000000.0;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 4) {
        if (rank == 0) {
            printf("Usage: %s <matrix_size> <max_iterations> <output_file>\n", argv[0]);
        }
        MPI_Finalize();
        exit(-1);
    }

    double starttime = gettime();
    double endtime = gettime();

    if(rank == 0) starttime = gettime();

    int rows = atoi(argv[1]);
    int cols = rows;
    int max_iterations = atoi(argv[2]);
    int hasChanged = 1;
    int lastI = 0;

    // seeding on main proc
    if (rank == 0) srand(100);

    // number of rows per process and handle remainder
    int base_rows = rows / size;
    int remaining_rows = rows % size;

    // rows per proc
    int local_rows = base_rows + (rank < remaining_rows ? 1 : 0);

    // allocate matrix with ghost arrays
    char *oldMatrix = (char *)malloc((local_rows + 2) * (cols + 2) * sizeof(char));
    char *newMatrix = (char *)malloc((local_rows + 2) * (cols + 2) * sizeof(char));

    // global matrix on main proc
    char *globalMatrix = NULL;
    if (rank == 0) {
        globalMatrix = (char *)malloc((rows + 2) * (cols + 2) * sizeof(char));
        initArr(rows, cols, globalMatrix);
        fillLife(rows, cols, globalMatrix);
        // printf("gen 0:\n");
        // printMatrix(rows, cols, globalMatrix);
    }

    // displacements and counts for scatter and gather
    int *sendcounts = malloc(size * sizeof(int));
    int *displs = malloc(size * sizeof(int));

    int offset = cols + 2; // Start after padding row
    for (int i = 0; i < size; i++) {
        int rows_for_process = base_rows + (i < remaining_rows ? 1 : 0);
        sendcounts[i] = rows_for_process * (cols + 2);
        displs[i] = offset;
        offset += sendcounts[i];
    }

    // Scatter rows to each process
    MPI_Scatterv(globalMatrix + (cols + 2), sendcounts, displs, MPI_CHAR, oldMatrix + (cols + 2), sendcounts[rank], MPI_CHAR, 0, MPI_COMM_WORLD);

    // simulation loop
    for (int i = 0; i < max_iterations && hasChanged; i++) {
        lastI = i;
        // exchange borders with neighboring processes
        if (rank > 0) {
            MPI_Sendrecv(oldMatrix + (cols + 2), cols + 2, MPI_CHAR, rank - 1, 0, oldMatrix, cols + 2, MPI_CHAR, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (rank < size - 1) {
            MPI_Sendrecv(oldMatrix + local_rows * (cols + 2), cols + 2, MPI_CHAR, rank + 1, 0, oldMatrix + (local_rows + 1) * (cols + 2), cols + 2, MPI_CHAR, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // run simulation for each local row
        hasChanged = simLife(rows, cols, oldMatrix, newMatrix, 1, local_rows);

        // this is to see if there has been a change across all proc
        // will kill sim if no change
        int globalChanged;
        MPI_Allreduce(&hasChanged, &globalChanged, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        hasChanged = globalChanged;

        // swap matrices for the next iteration
        char *temp = oldMatrix;
        oldMatrix = newMatrix;
        newMatrix = temp;

        if (!hasChanged) break;
    }

    // Gather the local parts back to the main process
    MPI_Gatherv(oldMatrix + (cols + 2), sendcounts[rank], MPI_CHAR, globalMatrix + (cols + 2), sendcounts, displs, MPI_CHAR, 0, MPI_COMM_WORLD);

    // prints only on rank 0;
    if (rank == 0) {
        endtime = gettime();
        FILE *outFile = fopen(argv[3], "w");
        fprintf(outFile, "======\nGEN %d\n======\n", lastI + 1);
        fprintfLife(rows, cols, globalMatrix, outFile);
        fprintf(outFile, "Time taken = %lf seconds\n", endtime - starttime);
        fclose(outFile);
        free(globalMatrix);
    }

    // frees matrices
    free(oldMatrix);
    free(newMatrix);
    free(sendcounts);
    free(displs);

    MPI_Finalize();
    return 0;
}
