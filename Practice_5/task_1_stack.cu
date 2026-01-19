%%writefile practice_work_5_stack.cu
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cstdlib>
#include <ctime>
#include <random>

using namespace std;

#define CUDA_CHECK(err) do { \
    cudaError_t e = (err); \
    if (e != cudaSuccess) { \
        cerr << "CUDA Error: " << cudaGetErrorString(e) \
             << " at line " << __LINE__ << endl; \
        exit(1); \
    } \
} while(0)

// ==============================
// Параметры
// ==============================
const int MAX_STACK_SIZE = 10000000;
const int BLOCK_SIZE = 256;
const int INITIAL_STACK_SIZE = 1000000;

// ==============================
// Глобальный стек на GPU
// ==============================
__device__ int d_stack[MAX_STACK_SIZE];
__device__ int d_top = 0;

// ==============================
// Инициализация стека
// ==============================
__global__ void init_stack_kernel(int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_stack[idx] = idx; // детерминированные значения
    }
    if (idx == 0) {
        d_top = n;
    }
}

// ==============================
// Push
// ==============================
__global__ void push_kernel(int* values, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int pos = atomicAdd(&d_top, 1);
    if (pos < MAX_STACK_SIZE) {
        d_stack[pos] = values[idx];
    } else {
        atomicSub(&d_top, 1);
    }
}

// ==============================
// Pop
// ==============================
__global__ void pop_kernel(int* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int pos = atomicSub(&d_top, 1);
    if (pos > 0) {
        output[idx] = d_stack[pos - 1];
    } else {
        atomicAdd(&d_top, 1);
        output[idx] = -1;
    }
}

// ==============================
// Печать стека (CPU)
// ==============================
void print_stack(const vector<int>& data, int top) {
    cout << "Stack size = " << top << "\n";

    cout << "First 10: ";
    for (int i = 0; i < min(10, top); i++)
        cout << data[i] << " ";
    cout << "\n";

    cout << "Last 10: ";
    for (int i = max(0, top - 10); i < top; i++)
        cout << data[i] << " ";
    cout << "\n\n";
}

// ==============================
// MAIN
// ==============================
int main() {
    cout << "=== Parallel Stack (LIFO) on GPU ===\n";

    const int N = 10000;

    // Данные для push
    vector<int> h_values(N);
    mt19937 gen(time(nullptr));
    uniform_int_distribution<int> dist(1, 1000000);
    for (int i = 0; i < N; ++i)
        h_values[i] = dist(gen);

    int *d_values, *d_output;
    CUDA_CHECK(cudaMalloc(&d_values, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_output, N * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_values, h_values.data(),
                          N * sizeof(int), cudaMemcpyHostToDevice));

    dim3 block(BLOCK_SIZE);
    dim3 grid_init((INITIAL_STACK_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // ==============================
    // ИНИЦИАЛИЗАЦИЯ СТЕКА
    // ==============================
    init_stack_kernel<<<grid_init, block>>>(INITIAL_STACK_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    vector<int> h_stack(MAX_STACK_SIZE);
    int h_top;

    CUDA_CHECK(cudaMemcpyFromSymbol(&h_top, d_top, sizeof(int)));
    CUDA_CHECK(cudaMemcpyFromSymbol(h_stack.data(), d_stack,
                                   MAX_STACK_SIZE * sizeof(int)));

    cout << "--- Before push ---\n";
    print_stack(h_stack, h_top);

    // ==============================
    // PUSH
    // ==============================
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    push_kernel<<<grid, block>>>(d_values, N);
    cudaEventRecord(stop);
    CUDA_CHECK(cudaDeviceSynchronize());

    float push_time;
    cudaEventElapsedTime(&push_time, start, stop);

    CUDA_CHECK(cudaMemcpyFromSymbol(&h_top, d_top, sizeof(int)));
    CUDA_CHECK(cudaMemcpyFromSymbol(h_stack.data(), d_stack,
                                   MAX_STACK_SIZE * sizeof(int)));

    cout << "--- After push ---\n";
    print_stack(h_stack, h_top);

    cout << "Expected stack size: " << INITIAL_STACK_SIZE + N << "\n";
    cout << "Actual stack size:   " << h_top << "\n\n";

    // ==============================
    // POP
    // ==============================
    cudaEventRecord(start);
    pop_kernel<<<grid, block>>>(d_output, N);
    cudaEventRecord(stop);
    CUDA_CHECK(cudaDeviceSynchronize());

    float pop_time;
    cudaEventElapsedTime(&pop_time, start, stop);

    CUDA_CHECK(cudaMemcpyFromSymbol(&h_top, d_top, sizeof(int)));
    CUDA_CHECK(cudaMemcpyFromSymbol(h_stack.data(), d_stack,
                                   MAX_STACK_SIZE * sizeof(int)));

    cout << "--- After pop ---\n";
    print_stack(h_stack, h_top);

    cout << "Push time: " << push_time << " ms\n";
    cout << "Pop time:  " << pop_time << " ms\n";

    cudaFree(d_values);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
