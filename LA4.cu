#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <chrono>

using namespace std::chrono;

#define RUNS 5
#define CPU_REPEAT 5

// ================= CUDA KERNEL =================
__global__ void vector_add(const float *input_a, const float *input_b, float *result_vector, int vector_size) {
    int thread_global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_stride = blockDim.x * gridDim.x;

    // Grid-stride loop (better GPU usage)
    for (int element_idx = thread_global_id; element_idx < vector_size; element_idx += grid_stride) {
        result_vector[element_idx] = input_a[element_idx] + input_b[element_idx];
    }
}

// ================= CPU SEQUENTIAL ADDITION =================
void cpu_vector(const float *input_a, const float *input_b, float *result_vector, int vector_size) {
    for (int i = 0; i < vector_size; i++) {
        // Slightly heavier computation to simulate workload
        float temp = input_a[i] + input_b[i];
        temp = temp * 1.0001f;
        temp = temp / 1.0001f;
        result_vector[i] = temp;
    }
}

int main() {
    printf("\n===== VECTOR ADD CUDA =====\n");

    cudaDeviceProp device_properties;
    cudaGetDeviceProperties(&device_properties, 0);
    printf("GPU: %s\n", device_properties.name);

    cudaFree(0); // Warm up CUDA context

    // Vector sizes to evaluate for benchmarking
    int vector_sizes[] = {10000, 13000, 14000, 17000};
    int total_tests = 4;

    FILE *results_file = fopen("la4.txt", "w");
    fprintf(results_file, "N,SEQ,PAR,SPEEDUP,EFFICIENCY,COST\n");

    for (int test_idx = 0; test_idx < total_tests; test_idx++) {
        int vector_size = vector_sizes[test_idx];
        size_t memory_bytes = vector_size * sizeof(float);

        // Host allocations (Pinned memory)
        float *host_vector_a, *host_vector_b, *host_vector_c;
        cudaMallocHost(&host_vector_a, memory_bytes);
        cudaMallocHost(&host_vector_b, memory_bytes);
        cudaMallocHost(&host_vector_c, memory_bytes);

        // Initialize host vectors with random data
        for (int i = 0; i < vector_size; i++) {
            host_vector_a[i] = rand() % 100;
            host_vector_b[i] = rand() % 100;
        }

        // Device allocations
        float *device_vector_a, *device_vector_b, *device_vector_c;
        cudaMalloc(&device_vector_a, memory_bytes);
        cudaMalloc(&device_vector_b, memory_bytes);
        cudaMalloc(&device_vector_c, memory_bytes);

        float total_gpu_time_ms = 0;
        float total_cpu_time_ms = 0;

        for (int run_idx = 0; run_idx < RUNS; run_idx++) {
            // GPU Timing Events
            cudaEvent_t gpu_start_event, gpu_stop_event;
            cudaEventCreate(&gpu_start_event);
            cudaEventCreate(&gpu_stop_event);

            cudaEventRecord(gpu_start_event);

            // Copy input data from Host to Device
            cudaMemcpy(device_vector_a, host_vector_a, memory_bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(device_vector_b, host_vector_b, memory_bytes, cudaMemcpyHostToDevice);

            // Block and Thread configurations
            int threads_per_block = 256;
            int num_blocks = 512; // Fixed high occupancy

            // Launch Kernel
            vector_add<<<num_blocks, threads_per_block>>>(device_vector_a, device_vector_b, device_vector_c, vector_size);

            // Copy result data from Device to Host
            cudaMemcpy(host_vector_c, device_vector_c, memory_bytes, cudaMemcpyDeviceToHost);

            cudaEventRecord(gpu_stop_event);
            cudaEventSynchronize(gpu_stop_event);

            float run_gpu_time_ms;
            cudaEventElapsedTime(&run_gpu_time_ms, gpu_start_event, gpu_stop_event);
            total_gpu_time_ms += run_gpu_time_ms;

            cudaEventDestroy(gpu_start_event);
            cudaEventDestroy(gpu_stop_event);

            // CPU Benchmark Timing
            float run_cpu_time_ms = 0;
            for (int i = 0; i < CPU_REPEAT; i++) {
                auto cpu_start_time = high_resolution_clock::now();
                cpu_vector(host_vector_a, host_vector_b, host_vector_c, vector_size);
                auto cpu_end_time = high_resolution_clock::now();
                run_cpu_time_ms += duration<float, std::milli>(cpu_end_time - cpu_start_time).count();
            }

            total_cpu_time_ms += run_cpu_time_ms / CPU_REPEAT;
        }

        // Metrics Calculation
        float avg_cpu_time_ms = total_cpu_time_ms / RUNS;
        float avg_gpu_time_ms = total_gpu_time_ms / RUNS;
        float measured_speedup = avg_cpu_time_ms / avg_gpu_time_ms;

        int num_blocks = 512;
        float parallel_efficiency = measured_speedup / num_blocks;
        float computational_cost = avg_gpu_time_ms * num_blocks;

        printf("N=%d | SEQ=%.3f ms | PAR=%.3f ms | SPEEDUP=%.3f | EFF=%.4f | COST=%.3f\n",
               vector_size, avg_cpu_time_ms, avg_gpu_time_ms, measured_speedup, parallel_efficiency, computational_cost);

        fprintf(results_file, "%d,%.3f,%.3f,%.3f,%.4f,%.3f\n",
                vector_size, avg_cpu_time_ms, avg_gpu_time_ms, measured_speedup, parallel_efficiency, computational_cost);

        // Deallocate device and host allocations
        cudaFree(device_vector_a);
        cudaFree(device_vector_b);
        cudaFree(device_vector_c);
        cudaFreeHost(host_vector_a);
        cudaFreeHost(host_vector_b);
        cudaFreeHost(host_vector_c);
    }

    fclose(results_file);
    printf("\nSaved to la4.txt\n");

    return 0;
}