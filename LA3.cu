#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <chrono>

using namespace std::chrono;

#define RUNS 5
#define CPU_REPEAT 10

// ================= CUDA KERNEL =================
// __global__ specifies a CUDA kernel function, launched by the CPU host to execute in parallel on the GPU device
__global__ void parallel_reduction_kernel(
    const float *device_input_data, 
    float *block_min_outputs, 
    float *block_max_outputs, 
    float *block_sum_outputs, 
    int num_elements
) {
    // __shared__ allocates high-speed, low-latency, on-chip shared memory shared by all threads in the same block
    __shared__ float shared_min[1024];
    __shared__ float shared_max[1024];
    __shared__ float shared_sum[1024];

    // threadIdx.x is a built-in CUDA variable representing the local 1D index of the thread within its block
    int local_thread_id = threadIdx.x;
    // blockIdx.x (block index) and blockDim.x (threads per block) are used to compute the global 1D index of the element
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

    // __syncthreads() is a block-level barrier synchronisation, waiting for all threads in the block to finish loading memory
    __syncthreads();

    // stride defines the exact distance between the two elements in shared memory that a single thread will compare or add together in a given iteration.
    // Perform reduction in shared memory using a binary tree structure
    // stride starts at blockDim.x / 2 and is halved at each step using right shift (stride >>= 1) until it is 0
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        // Only threads with indexes less than the current stride are active in this step
        if (local_thread_id < stride) {
            shared_min[local_thread_id] = min(shared_min[local_thread_id], shared_min[local_thread_id + stride]);
            shared_max[local_thread_id] = max(shared_max[local_thread_id], shared_max[local_thread_id + stride]);
            shared_sum[local_thread_id] += shared_sum[local_thread_id + stride];
        }
        // Sync threads at the end of each tree level to ensure all thread writes are visible for the next stride level
        __syncthreads();
    }

    // Write the result of this block's reduction to global memory
    // Thread 0 represents the final reduced value of the entire block and writes to the block index blockIdx.x
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

    // cudaDeviceProp is a CUDA struct storing properties and physical parameters of the target GPU device
    cudaDeviceProp device_properties;
    // cudaGetDeviceProperties fills the device_properties struct with information about GPU device 0
    cudaGetDeviceProperties(&device_properties, 0);

    // MultiProcessor count represents the Streaming Multiprocessor (SM) count on the GPU
    int estimated_cuda_cores = device_properties.multiProcessorCount * 128;

    printf("GPU = %s\n", device_properties.name);
    printf("CUDA Cores (estimated) = %d\n\n", estimated_cuda_cores);

    // cudaFree(0) warms up the lazy-loaded CUDA runtime context to prevent initialization latency from biasing timing
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

            // cudaMalloc allocates linear memory on the GPU device's global memory heap
            cudaMalloc(&device_input_array, memory_bytes);
            cudaMalloc(&device_block_min_outputs, sizeof(float) * num_blocks);
            cudaMalloc(&device_block_max_outputs, sizeof(float) * num_blocks);
            cudaMalloc(&device_block_sum_outputs, sizeof(float) * num_blocks);

            // cudaMemcpy copies data from Host-to-Device (CPU host memory to GPU device memory)
            cudaMemcpy(device_input_array, host_input_array, memory_bytes, cudaMemcpyHostToDevice);

            // ================= GPU BENCHMARK =================
            // cudaEvent_t represents a high-resolution CUDA event object for device-side timing/synchronization
            cudaEvent_t gpu_start_event, gpu_stop_event;
            // cudaEventCreate instantiates and allocates resources for the timing events
            cudaEventCreate(&gpu_start_event);
            cudaEventCreate(&gpu_stop_event);

            // cudaEventRecord registers the start event marker in the GPU execution queue
            cudaEventRecord(gpu_start_event);

            // Launch the kernel multiple times to simulate realistic scaling
            for (int launch_count = 0; launch_count < 5; launch_count++) {
                // Kernel call using execution configuration triple angle brackets <<<num_blocks, threads_per_block>>>
                parallel_reduction_kernel<<<num_blocks, threads_per_block>>>(
                    device_input_array, 
                    device_block_min_outputs, 
                    device_block_max_outputs, 
                    device_block_sum_outputs, 
                    current_array_size
                );
            }

            // cudaDeviceSynchronize blocks host CPU execution until all enqueued GPU kernels have fully finished execution
            cudaDeviceSynchronize();

            // cudaEventRecord registers the stop event marker in the GPU execution queue
            cudaEventRecord(gpu_stop_event);
            // cudaEventSynchronize blocks CPU host thread until the stop event has been recorded by the GPU
            cudaEventSynchronize(gpu_stop_event);

            float gpu_parallel_time_ms;
            // cudaEventElapsedTime calculates the elapsed time in milliseconds between two recorded events
            cudaEventElapsedTime(&gpu_parallel_time_ms, gpu_start_event, gpu_stop_event);

            // cudaEventDestroy releases memory and resources allocated for the CUDA events
            cudaEventDestroy(gpu_start_event);
            cudaEventDestroy(gpu_stop_event);

            // cudaFree deallocates memory blocks previously allocated on the GPU device's global memory heap
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