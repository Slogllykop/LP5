// g++ -fopenmp bfs.cpp -o a; ./a

#include <iostream>
#include <vector>
#include <queue>
#include <fstream>
#include <omp.h>

using namespace std;

class Graph {
public:
    int V;
    vector<vector<int>> adj;

    Graph(int V) {
        this->V = V;
        adj.resize(V);
    }

    void addEdge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u);
    }


    // ---------------- SEQ BFS ----------------
    void seqBFS(int start) {
        vector<char> visited(V, 0);
        queue<int> q;

        visited[start] = 1;
        q.push(start);

        while (!q.empty()) {
            int node = q.front();
            q.pop();

            for (int n : adj[node]) {
                if (!visited[n]) {
                    visited[n] = 1;
                    q.push(n);
                }
            }
        }
    }

    // ---------------- PAR BFS ----------------
    void parBFS(int start) {
        vector<char> visited(V, 0);
        vector<int> frontier{start};

        visited[start] = 1;

        while (!frontier.empty()) {
            vector<int> next;

            // OpenMP parallel region: Spawns a team of threads to execute the block concurrently.
            // Each thread will maintain its own thread-local 'local' vector on its private stack.
            #pragma omp parallel
            {
                vector<int> local;

                // OpenMP loop parallelization: Dynamically distributes loop iterations among active threads.
                // schedule(dynamic) dynamically chunks work (default chunk size 1), which is ideal for graph
                // traversals since node degrees (and thus processing workloads) vary significantly.
                #pragma omp for schedule(dynamic)

                for (int i = 0; i < frontier.size(); i++) {

                    int node = frontier[i];

                    for (int n : adj[node]) {

                        if (!visited[n]) {

                            bool added = false;

                            // OpenMP critical section: Enforces mutual exclusion so only one thread executes this block at a time.
                            // Thread-safely performs a double-checked visited status check before setting visited[n] = 1
                            // to prevent multiple threads from concurrently discovering and processing the same node.
                            #pragma omp critical
                            {
                                if (!visited[n]) {
                                    visited[n] = 1;
                                    added = true;
                                }
                            }

                        if (added)
                              local.push_back(n);


                        }


                    }
                }


                // OpenMP critical section: Serializes concurrent writes to the shared 'next' frontier vector.
                // Ensures each thread safe-merges its thread-local discoveries into the global frontier buffer.
                #pragma omp critical
                next.insert(next.end(), local.begin(), local.end());

            }

            frontier.swap(next);

        }
    }
};

// ---------------- MAIN ----------------
int main() {
    ofstream file("la1.txt");

    file << "N,SEQ,PAR,SPEEDUP,EFF,COST\n";

    //  FORCE THREADS
    int cores = 4;
    omp_set_num_threads(cores);

    cout << "Using " << cores << " threads (limited for crossover detection)\n";

    for (int N = 100; N <= 3000; N += 500) {

        Graph g(N);

        //  scalable density
        for (int i = 0; i < N; i++) {
            for (int j = i + 1; j < min(N, i + (N / 20)); j++) {
                g.addEdge(i, j);
            }
        }

        double t1, t2;

        // ================= BFS =================
        t1 = omp_get_wtime();
        g.seqBFS(0);
        t2 = omp_get_wtime();
        double bfs_seq = t2 - t1;

        t1 = omp_get_wtime();
        g.parBFS(0);
        t2 = omp_get_wtime();
        double bfs_par = t2 - t1;

        double bfs_speed = bfs_seq / bfs_par;
        double bfs_eff = bfs_speed / cores;
        double bfs_cost = bfs_par * cores;

        // ================= PRINT =================
        cout << "\n====================================\n";
        cout << "N = " << N << endl;

        cout << "\n--- BFS ---\n";
        cout << "Sequential Time : " << bfs_seq << endl;
        cout << "Parallel Time   : " << bfs_par << endl;
        cout << "Speedup         : " << bfs_speed << endl;
        cout << "Efficiency      : " << bfs_eff << endl;
        cout << "Cost            : " << bfs_cost << endl;

        // ================= FILE =================
        file << N << ","
             << bfs_seq << "," 
             << bfs_par << "," 
             << bfs_speed << "," 
             << bfs_eff << "," 
             << bfs_cost << "\n";



    }


    file.close();
    cout << "\nSaved to la1.txt\n";
}
