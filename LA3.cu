#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <chrono>

using namespace std::chrono;

#define RUNS 5
#define CPU_REPEAT 10

// ================= CUDA KERNEL =================
__global__ void parallel_reduction_kernel(
    const float *device_input_data, 
    float *block_min_outputs, 
    float *block_max_outputs, 
    float *block_sum_outputs, 
    int num_elements
) {
    __shared__ float shared_min[1024];
    __shared__ float shared_max[1024];
    __shared__ float shared_sum[1024];

    int local_thread_id = threadIdx.x;
    int global_element_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load elements into shared memory, or pad with neutral elements if out of bounds
    if (global_element_idx < num_elements) {
        shared_min[local_thread_id] = device_input_data[global_element_idx];
        shared_max[local_thread_id] = device_input_data[global_element_idx];
        shared_sum[local_thread_id] = device_input_data[global_element_idx];
    } else {
        shared_min[local_thread_id] = 1e9f;  // Positive infinity/neutral value for min
        shared_max[local_thread_id] = -1e9f; // Negative infinity/neutral value for max
        shared_sum[local_thread_id] = 0.0f;  // Zero/neutral value for sum
    }

    __syncthreads();

    // Perform reduction in shared memory using a binary tree structure
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (local_thread_id < stride) {
            shared_min[local_thread_id] = min(shared_min[local_thread_id], shared_min[local_thread_id + stride]);
            shared_max[local_thread_id] = max(shared_max[local_thread_id], shared_max[local_thread_id + stride]);
            shared_sum[local_thread_id] += shared_sum[local_thread_id + stride];
        }
        __syncthreads();
    }

    // Write the result of this block's reduction to global memory
    if (local_thread_id == 0) {
        block_min_outputs[blockIdx.x] = shared_min[0];
        block_max_outputs[blockIdx.x] = shared_max[0];
        block_sum_outputs[blockIdx.x] = shared_sum[0];
    }
}

// ================= CPU =================
void cpu_compute_sequential(
    const float *host_input_data, 
    float *sequential_min_result, 
    float *sequential_max_result, 
    float *sequential_sum_result, 
    int num_elements
) {
    *sequential_min_result = host_input_data[0];
    *sequential_max_result = host_input_data[0];
    *sequential_sum_result = 0.0f;

    for (int i = 0; i < num_elements; i++) {
        if (host_input_data[i] < *sequential_min_result) {
            *sequential_min_result = host_input_data[i];
        }
        if (host_input_data[i] > *sequential_max_result) {
            *sequential_max_result = host_input_data[i];
        }
        *sequential_sum_result += host_input_data[i];
    }
}

// ================= MAIN =================
int main() {

    printf("\n===== CUDA Multi-N Multi-Run GRAPH DATA (UPDATED) =====\n");

    cudaDeviceProp device_properties;
    cudaGetDeviceProperties(&device_properties, 0);

    int estimated_cuda_cores = device_properties.multiProcessorCount * 128;

    printf("GPU = %s\n", device_properties.name);
    printf("CUDA Cores (estimated) = %d\n\n", estimated_cuda_cores);

    cudaFree(0); // warm-up device context

    // Array sizes to evaluate for benchmarking
    int array_sizes_to_test[] = { 8000, 9000, 10000, 11000, 13000, 14000, 15000, 16000, 17000, 18000};
    int total_tests = 10;

    FILE *results_csv_file = fopen("reduction_result.csv", "w");
    fprintf(results_csv_file, "N,RUN,SEQ_MS,PAR_MS,SPEEDUP,EFFICIENCY,COST\n");

    for (int test_idx = 0; test_idx < total_tests; test_idx++) {

        int current_array_size = array_sizes_to_test[test_idx];
        printf("\nRunning with N = %d elements\n", current_array_size);

        size_t memory_bytes = current_array_size * sizeof(float);
        float *host_input_array = (float*)malloc(memory_bytes);

        int threads_per_block = 1024;
        int num_blocks = (current_array_size + threads_per_block - 1) / threads_per_block;

        for (int run_idx = 0; run_idx < RUNS; run_idx++) {

            // Populate host array with random floating-point values
            for (int i = 0; i < current_array_size; i++) {
                host_input_array[i] = rand() % 100;
            }

            float *device_input_array;
            float *device_block_min_outputs;
            float *device_block_max_outputs;
            float *device_block_sum_outputs;

            cudaMalloc(&device_input_array, memory_bytes);
            cudaMalloc(&device_block_min_outputs, sizeof(float) * num_blocks);
            cudaMalloc(&device_block_max_outputs, sizeof(float) * num_blocks);
            cudaMalloc(&device_block_sum_outputs, sizeof(float) * num_blocks);

            cudaMemcpy(device_input_array, host_input_array, memory_bytes, cudaMemcpyHostToDevice);

            // ================= GPU BENCHMARK =================
            cudaEvent_t gpu_start_event, gpu_stop_event;
            cudaEventCreate(&gpu_start_event);
            cudaEventCreate(&gpu_stop_event);

            cudaEventRecord(gpu_start_event);

            // Launch the kernel multiple times to simulate realistic scaling
            for (int launch_count = 0; launch_count < 5; launch_count++) {
                parallel_reduction_kernel<<<num_blocks, threads_per_block>>>(
                    device_input_array, 
                    device_block_min_outputs, 
                    device_block_max_outputs, 
                    device_block_sum_outputs, 
                    current_array_size
                );
            }

            cudaDeviceSynchronize();

            cudaEventRecord(gpu_stop_event);
            cudaEventSynchronize(gpu_stop_event);

            float gpu_parallel_time_ms;
            cudaEventElapsedTime(&gpu_parallel_time_ms, gpu_start_event, gpu_stop_event);

            cudaEventDestroy(gpu_start_event);
            cudaEventDestroy(gpu_stop_event);

            cudaFree(device_input_array);
            cudaFree(device_block_min_outputs);
            cudaFree(device_block_max_outputs);
            cudaFree(device_block_sum_outputs);

            // ================= CPU BENCHMARK =================
            float total_cpu_time_ms = 0;

            for (int i = 0; i < CPU_REPEAT; i++) {
                auto cpu_start_time = high_resolution_clock::now();

                float cpu_min_val, cpu_max_val, cpu_sum_val;
                cpu_compute_sequential(host_input_array, &cpu_min_val, &cpu_max_val, &cpu_sum_val, current_array_size);

                auto cpu_end_time = high_resolution_clock::now();

                total_cpu_time_ms += duration<float, std::milli>(cpu_end_time - cpu_start_time).count();
            }

            float cpu_average_sequential_time_ms = total_cpu_time_ms / CPU_REPEAT;

            float measured_speedup = cpu_average_sequential_time_ms / gpu_parallel_time_ms;
            float parallel_efficiency = measured_speedup / estimated_cuda_cores;
            float computational_cost = gpu_parallel_time_ms * estimated_cuda_cores;

            // ================= SAVE RESULTS =================
            fprintf(results_csv_file, "%d,%d,%f,%f,%f,%f,%f\n",
                    current_array_size, run_idx, cpu_average_sequential_time_ms, gpu_parallel_time_ms, measured_speedup, parallel_efficiency, computational_cost);

            printf("Run %d | SEQ=%f ms | PAR=%f ms | SPEEDUP=%f | EFF=%f | COST=%f\n",
                   run_idx, cpu_average_sequential_time_ms, gpu_parallel_time_ms, measured_speedup, parallel_efficiency, computational_cost);
        }

        free(host_input_array);
    }

    fclose(results_csv_file);

    printf("\nRESULT SAVED IN reduction_result.csv\n");

    return 0;
}

// GTX 1050 / 1060 / 1070 / 1080 (Pascal) -> 128
// GTX 1650 / 1660 / RTX 2060 / 2070 / 2080 (Turing) -> 64
// RTX 3060 / 3070 / 3080 (Ampere) -> 128
// RTX 4060 / 4070 / 4080 (Ada) -> 128
// Titan X (Maxwell) / Titan X (Pascal) / Titan Xp -> 128
// Titan V / Titan RTX / Quadro RTX series (e.g., RTX 8000, RTX 5000) -> 64