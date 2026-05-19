#include <iostream>
#include <omp.h>
#include <vector>
#include <cstdlib>
#include <fstream>
#include <ctime>

using namespace std;

// Preprocessor directive to define a macro constant 'TESTS' representing the number of input size test cases.
#define TESTS 6
int N_values[TESTS] = {100, 300, 500, 700, 900, 1000};

// ================= UTILITY =================
void generateArray(vector<int> &arr, int N) {
    for (int i = 0; i < N; i++)
        arr[i] = rand() % 10000;
}

void copyArray(vector<int> &src, vector<int> &dest, int N) {
    for (int i = 0; i < N; i++)
        dest[i] = src[i];
}

// ================= BUBBLE SORT =================
void seq_bubble(vector<int> &arr, int N) {
    for (int i = 0; i < N - 1; i++)
        for (int j = 0; j < N - i - 1; j++)
            if (arr[j] > arr[j + 1])
                swap(arr[j], arr[j + 1]);
}


void par_bubble(vector<int> &arr, int N) {

    for (int phase = 0; phase < N; phase++) {

        // ================= EVEN PHASE =================
        if (phase % 2 == 0) {

            // OpenMP compiler directive to parallelize the following for-loop.
            // It instructs the compiler to generate code that distributes the loop iterations across multiple threads.
            #pragma omp parallel for
            for (int i = 0; i < N - 1; i += 2) {
                if (arr[i] > arr[i + 1]) {
                    swap(arr[i], arr[i + 1]);
                }
            }

        }
        // ================= ODD PHASE =================
        else {

            // OpenMP compiler directive to parallelize the following for-loop.
            // It instructs the compiler to generate code that distributes the loop iterations across multiple threads.
            #pragma omp parallel for
            for (int i = 1; i < N - 1; i += 2) {
                if (arr[i] > arr[i + 1]) {
                    swap(arr[i], arr[i + 1]);
                }
            }
        }
    }
}


// ================= MAIN =================
int main() {

    srand(time(NULL));
    int cores = omp_get_max_threads();

    ofstream file("la2.txt");

    // Added COST column
    file << "N,SEQ,PAR,SPEEDUP,EFFICIENCY,COST,CORES\n";

    for (int t = 0; t < TESTS; t++) {

        int N = N_values[t];
        vector<int> arr(N), temp(N);

        generateArray(arr, N);

        double seq, par, speed, eff, cost;

        cout << "\n============================\n";
        cout << "N = " << N << endl;

        // ===== BUBBLE =====
        copyArray(arr, temp, N);
        seq = omp_get_wtime();
        seq_bubble(temp, N);
        seq = omp_get_wtime() - seq;

        copyArray(arr, temp, N);
        par = omp_get_wtime();
        par_bubble(temp, N);
        par = omp_get_wtime() - par;

        speed = seq / par;
        eff = speed / cores;

        cost = par * cores;

        cout << "BUBBLE ->"
             << " Seq=" << seq
             << " Par=" << par
             << " Speedup=" << speed
             << " Efficiency=" << eff
             << " Cost=" << cost << endl;

        file << N << "," 
             << seq << "," 
             << par << "," 
             << speed << "," 
             << eff << "," 
             << cost << "," 
             << cores << "\n";
    }

    file.close();

    cout << "\nResults saved to la2.txt\n";
    return 0;
}