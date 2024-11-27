/*
Name: Chris Phornroekngam
Email: cgphornroekngam@crimson.ua.edu
Course Section: CS 481
Homework #: 1
Mac Compilation:    clang -Xclang -fopenmp -L/opt/homebrew/opt/libomp/lib -I/opt/homebrew/opt/libomp/include -lomp hw3.c -o hw3
                    OR
                    gcc-14 -fopenmp -Wall -O -o hw3 hw3.c
ASC Compilation:    ./asc-run.sh
                    OR
                    icx -g -Wall -o hw3 hw3.c -fopenmp
Usage: ./hw1 <matrix_size> <max_itgccerations>
              
*/

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <time.h>
#include <omp.h>

void fprintfLife(int r, int c, char **arr, FILE* file) {
    for (int i = 1; i < r+1; i++)
    {
        for (int j = 1; j < c+1; j++)
        {
            fprintf(file, "%c ", arr[i][j]);
        }
        fprintf(file, "\n");
    }
}

double gettime(void) {
  struct timeval tval;

  gettimeofday(&tval, NULL);

  return( (double)tval.tv_sec + (double)tval.tv_usec/1000000.0 );
}

// inits all arrays to 0
void initArr(int r, int c, char **arr) {
    int i, j;
    for (i = 0; i < r+2; i++) {
        for (j = 0; j < c+2; j++)
        {

            arr[i][j] = '0';
        }
            
    }
}

// generates a random starting life matrix
void fillLife(int r, int c, char **arr) {
    int i, j;
    for (i = 1; i < r+1; i++) {
        for (j = 1; j < c+1; j++)
        {
            int temp = rand() % 2;
            if(temp) {
                arr[i][j] = '1';
            }
            else {
                arr[i][j] = '0';
            }
        }
            
    }
}

// copies matrix from old to new arr
void copyMatrix(int r, int c, char **oldArr, char **newArr, int threads) {
    int i, j;
    // parallel region for copying
    // #pragma omp parallel for private(i, j) num_threads(threads)
    for (i = 1; i < r+1; i++) {
        for (j = 1; j < c+1; j++)
        {
            newArr[i][j] = oldArr[i][j];
        }
            
    }
}

// sims life according to game rules and returns 1 if changed
int simLife(int r, int c, char **oldArr, char **newArr, int threads) {
    int changed = 0;

    #pragma omp parallel default(none) shared(oldArr, newArr, r, c, threads, changed) num_threads(threads)
    {
        int tid = omp_get_thread_num();
        int num_threads = omp_get_num_threads();

        // calculate the start and end row for this thread
        int start_row = (r * tid) / num_threads + 1;  // +1 to account for the offset
        int end_row = (r * (tid + 1)) / num_threads + 1; // Range up to r+1

        for (int iR = start_row; iR < end_row; iR++) {
            for (int jC = 1; jC < c + 1; jC++) {
                int aliveNeighbors = 0;

                // counting alive neighbors
                // top row
                if (oldArr[iR-1][jC-1] == '1') aliveNeighbors++;
                if (oldArr[iR-1][jC] == '1') aliveNeighbors++;
                if (oldArr[iR-1][jC+1] == '1') aliveNeighbors++;

                // mid row
                if (oldArr[iR][jC-1] == '1') aliveNeighbors++;
                if (oldArr[iR][jC+1] == '1') aliveNeighbors++;

                // bot row
                if (oldArr[iR+1][jC-1] == '1') aliveNeighbors++;
                if (oldArr[iR+1][jC] == '1') aliveNeighbors++;
                if (oldArr[iR+1][jC+1] == '1') aliveNeighbors++;

                // alive case
                if (oldArr[iR][jC] == '1') {
                    if (aliveNeighbors <= 1 || aliveNeighbors >= 4) {
                        newArr[iR][jC] = '0';
                        changed = 1;
                    }
                }
                // dead case
                else if (oldArr[iR][jC] == '0') {
                    if (aliveNeighbors == 3) {
                        newArr[iR][jC] = '1';
                        changed = 1;
                    }
                }
            }
        }
    }

    return changed;
}

int main(int argc, char **argv) {

    if (argc != 5) {
        printf("Usage: %s <matrix_size> <max_iterations> <num_threads> <output_file>\n", argv[0]);
        exit(-1);
    }
    int numThreads = atoi(argv[3]);
    

    double starttime = gettime();

    // seeding time
    srand(time(NULL));
    // srand(atoi(argv[3]));
    srand(100);


    int rows = atoi(argv[1]);
    int cols = rows;
    int i;
    // allocate matrices
    char **lifeMatrix = (char**)malloc((rows+2)*sizeof(char*));
    for (i = 0; i < rows+2; i++) {
        lifeMatrix[i] = (char*)malloc((cols+2) * sizeof(char));
    }

    char **oldMatrix = (char**)malloc((rows+2)*sizeof(char*));
    for (i = 0; i < rows+2; i++) {
        oldMatrix[i] = (char*)malloc((cols+2) * sizeof(char));
    }




    
    initArr(rows, cols, oldMatrix);
    fillLife(rows, cols, oldMatrix);
    
    copyMatrix(rows, cols, oldMatrix, lifeMatrix, numThreads);
    // printf("\n======\nGEN 0\n======\n");
    // printLife(rows, cols, lifeMatrix);

    int lastI = 0;
    int hasChanged = 1;
    for (i = 0; i < atoi(argv[2]); i++) {
        
        hasChanged = simLife(rows, cols, oldMatrix, lifeMatrix, numThreads);
        lastI = i;

        // debug prints
        // printf("\n======\nGEN %d\n======\n", i+1);
        // printLife(rows, cols, lifeMatrix);

        // checks to ensure we don't get stuck in endless loop
        if(!hasChanged) {
            break;
        }

        copyMatrix(rows, cols, lifeMatrix, oldMatrix, numThreads);

    }
    

    double endtime = gettime();

    // printf("Time taken = %lf seconds\n", endtime-starttime);

    FILE* outFile = fopen(argv[4], "w");

    fprintf(outFile, "\n======\nGEN %d\n======\n", lastI+1);
    fprintfLife(rows, cols, lifeMatrix, outFile);
    if(hasChanged == 0) {
        fprintf(outFile, "No changes b/w GEN %d and GEN %d\nLast viable GEN is GEN %d\n", lastI, lastI+1, lastI);
    }
    fprintf(outFile, "Time taken = %lf seconds\n", endtime-starttime);



    // freeing dynamic arrays
    for (i = 0; i < rows; i++)
        free(lifeMatrix[i]);
    free(lifeMatrix);

    for (i = 0; i < rows; i++)
        free(oldMatrix[i]);
    free(oldMatrix);
    fclose(outFile);

    return 0;
}
