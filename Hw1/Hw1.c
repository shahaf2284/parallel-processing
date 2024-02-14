//
// Created by Iris Eting on 09/02/2024.
//
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"

// Function to generate unique seeds for random number generation
void generateUniqueSeeds(int rank, int* seeds, int numProcesses);

// Function to perform Monte Carlo simulation
void performMonteCarloSimulation(int rank, int* seeds, int numProcesses, int attempts, int* successes);

// Function to print results
void printResults(int rank, int master, int totalSuccesses, int attempts, double startTime);

int main(int argc, char *argv[]) {
    // MPI variables
    int master = 0;
    int rank;
    int numProcesses;

    // Number of Monte Carlo attempts
    int attempts = 900000000;

    // Count of successful attempts across all processes
    int successes = 0;

    // Constant for PI with 25 decimal places
    double PI25DT = 3.141592653589793238462643;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Allocate memory for unique seeds
    int* uniqueSeeds = (int*)malloc(sizeof(int) * numProcesses);

    // Generate unique seeds for random number generation
    generateUniqueSeeds(rank, uniqueSeeds, numProcesses);

    // Record start time
    double startTime = MPI_Wtime();

    // Perform Monte Carlo simulation
    performMonteCarloSimulation(rank, uniqueSeeds, numProcesses, attempts, &successes);

    // Reduce successes across all processes
    int totalSuccesses = 0;
    MPI_Reduce(&successes, &totalSuccesses, 1, MPI_INT, MPI_SUM, master, MPI_COMM_WORLD);

    // Print results
    printResults(rank, master, totalSuccesses, attempts, startTime);

    // Free allocated memory
    free(uniqueSeeds);

    // Finalize MPI
    MPI_Finalize();

    return 0;
}

// Function to generate unique seeds for random number generation
void generateUniqueSeeds(int rank, int* seeds, int numProcesses) {
    // Only rank 0 generates unique seeds
    if (rank == 0) {
        for (int i = 0; i < numProcesses; i++) {
            seeds[i] = rand();
        }
    }

    // Broadcast generated seeds to all processes
    MPI_Bcast(seeds, numProcesses, MPI_INT, 0, MPI_COMM_WORLD);
}

// Function to perform Monte Carlo simulation
void performMonteCarloSimulation(int rank, int* seeds, int numProcesses, int attempts, int* successes) {
    // Seed the random number generator with the unique seed for each process
    srand(seeds[rank]);

    // Perform Monte Carlo simulation for the assigned number of attempts
    for (int i = 0; i < (attempts / numProcesses); i++) {
        double x = (double)rand() / RAND_MAX * 2 - 1;
        double y = (double)rand() / RAND_MAX * 2 - 1;
        double distance = x * x + y * y;

        // Check if the point is inside the unit circle
        if (distance < 1) {
            (*successes)++;
        }
    }
}

// Function to print results
void printResults(int rank, int master, int totalSuccesses, int attempts, double startTime) {
    // Only the master process prints the final results
    if (rank == 0) {
        double endTime = MPI_Wtime();
        double PI25DT = 3.141592653589793238462643;
        double pi = 4 * ((double)totalSuccesses / attempts);

        printf("Total HITS across all processes = %d\n", totalSuccesses);
        printf("pi is approximately %.16f, Error is %.16f\n", pi, fabs(pi - PI25DT));
        printf("Computation time = %f seconds\n", endTime - startTime);
        fflush(stdout);
    }
}
