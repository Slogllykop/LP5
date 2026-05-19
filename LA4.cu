#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <chrono>

using namespace std::chrono;

#define RUNS 5
#define CPU_REPEAT 5

// ================= CUDA KERNEL =================
// __global__ is a CUDA execution space specifier. It indicates that vector_add is a kernel function
// that is called from the CPU (Host) and runs asynchronously on the GPU (Device) across multiple threads.
__global__ void vector_add(
    const float *input_a, 
    const float *input_b, 
    float *result_vector, 
    int vector_size
) {
    // blockIdx, blockDim, and threadIdx are built-in, read-only CUDA variables available in device space.
    // blockIdx.x: The 1D index of the block within the grid.
    // blockDim.x: The dimension (number of threads) in each thread block.
    // threadIdx.x: The 1D index of the current thread within its thread block.
    // This line computes the global thread ID: the unique, grid-wide index for the current active thread.
    // It is critical to map each thread to its unique position in the parallel computational workload.
    int thread_global_id = blockIdx.x * blockDim.x + threadIdx.x;

    // gridDim.x: The total number of blocks in the grid.
    // Calculates the total number of active threads launched in the entire grid.
    // Essential for grid-stride looping, allowing threads to leap forward and process the next chunk of elements.
    int grid_stride = blockDim.x * gridDim.x;

    // Grid-stride loop pattern:
    // Starts processing at thread_global_id, and moves by grid_stride in each iteration.
    // This ensures hardware-agnostic grid configurations, thread reuse, and absolute bounds safety.
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

    // cudaDeviceProp is a runtime struct holding GPU capabilities (e.g. name, compute capability, SM count).
    cudaDeviceProp device_properties;
    
    // Retrieves hardware properties for GPU index 0. Essential for querying hardware specifications dynamically.
    cudaGetDeviceProperties(&device_properties, 0);
    printf("GPU: %s\n", device_properties.name);

    // cudaFree(0) triggers the driver to initialize the CUDA context on GPU device 0 immediately.
    // This is an essential "warmup" call to ensure context initialization overhead is paid upfront,
    // rather than skewing the subsequent execution timers during the first benchmark run.
    cudaFree(0); 

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
        
        // cudaMallocHost allocates page-locked (pinned) CPU memory. Pinned host memory prevents the OS 
        // from swapping pages to disk, enabling the GPU to use fast Direct Memory Access (DMA) 
        // to transfer data over the PCIe bus much faster than standard pageable memory.
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
        
        // cudaMalloc allocates persistent global memory directly on the GPU device's onboard RAM (VRAM).
        // This is critical because GPU cores cannot directly read or write to normal CPU system RAM.
        cudaMalloc(&device_vector_a, memory_bytes);
        cudaMalloc(&device_vector_b, memory_bytes);
        cudaMalloc(&device_vector_c, memory_bytes);

        float total_gpu_time_ms = 0;
        float total_cpu_time_ms = 0;

        for (int run_idx = 0; run_idx < RUNS; run_idx++) {
            // cudaEvent_t represents opaque CUDA event handles used for timing and synchronization on the GPU.
            cudaEvent_t gpu_start_event, gpu_stop_event;
            
            // Creates hardware-level events on the GPU. Essential for high-precision, low-overhead device timing.
            cudaEventCreate(&gpu_start_event);
            cudaEventCreate(&gpu_stop_event);

            // Records a start timestamp directly in the GPU command stream.
            // Critical because GPU operations are asynchronous; using CPU timers here would measure instantly instead of waiting.
            cudaEventRecord(gpu_start_event);

            // Copies array data from Host (CPU) pinned memory to Device (GPU) global memory.
            // cudaMemcpyHostToDevice specifies the direction. Essential to populate data on the GPU before calculations.
            cudaMemcpy(device_vector_a, host_vector_a, memory_bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(device_vector_b, host_vector_b, memory_bytes, cudaMemcpyHostToDevice);

            // Block and Thread configurations:
            // - threads_per_block = 256: The standard "sweet spot" block size in CUDA. It is a multiple of the 32-thread warp
            //   size to avoid execution waste, hiding memory latency while balancing register/shared-memory limits per block.
            int threads_per_block = 256;
            // - num_blocks = 512: A fixed grid size of 512 blocks. This launches enough blocks to fully saturate and occupy
            //   all Streaming Multiprocessors (SMs) on the GPU, while the grid-stride loop handles any remaining data.
            int num_blocks = 512;

            // Launch Kernel using <<<blocks, threads>>> syntax.
            // Schedules the vector_add kernel to run with 512 parallel thread blocks of 256 threads each.
            // The <<< >>> is the CUDA execution configuration syntax for initiating GPU computing grids.
            vector_add<<<num_blocks, threads_per_block>>>(
                device_vector_a, 
                device_vector_b, 
                device_vector_c, 
                vector_size
            );

            // Copies the result array from Device (GPU) memory back to Host (CPU) memory.
            // cudaMemcpyDeviceToHost specifies the direction. Essential to retrieve computed values back to CPU space.
            cudaMemcpy(host_vector_c, device_vector_c, memory_bytes, cudaMemcpyDeviceToHost);

            // Records a stop timestamp in the GPU command stream.
            cudaEventRecord(gpu_stop_event);
            
            // Blocks Host (CPU) execution until all operations prior to gpu_stop_event on the GPU finish.
            // Critical because CUDA kernels and memcpys are asynchronous; without sync, the CPU would read timing prematurely.
            cudaEventSynchronize(gpu_stop_event);

            float run_gpu_time_ms;
            
            // Calculates the elapsed time (in ms) between the recorded start and stop events on the GPU.
            // Essential to get high-precision timing at the microsecond level directly from the GPU queue.
            cudaEventElapsedTime(&run_gpu_time_ms, gpu_start_event, gpu_stop_event);
            total_gpu_time_ms += run_gpu_time_ms;

            // Destroys the CUDA events to free up underlying GPU context tracking resources and prevent memory leaks.
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

        // cudaFree deallocates global device memory previously allocated on the GPU via cudaMalloc.
        // Critical to prevent device VRAM leaks, which lead to Out of Memory (OOM) failures.
        cudaFree(device_vector_a);
        cudaFree(device_vector_b);
        cudaFree(device_vector_c);
        
        // cudaFreeHost deallocates host page-locked (pinned) memory previously allocated via cudaMallocHost.
        // Critical to release locked physical memory and avoid leaking system RAM.
        cudaFreeHost(host_vector_a);
        cudaFreeHost(host_vector_b);
        cudaFreeHost(host_vector_c);
    }

    fclose(results_file);
    printf("\nSaved to la4.txt\n");

    return 0;
}