%%writefile practice_work_5_queue.cu
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
const int MAX_QUEUE_SIZE = 10000000;
const int BLOCK_SIZE = 256;
const int INITIAL_QUEUE_SIZE = 1000000;

// ==============================
// Глобальная очередь на GPU
// ==============================
__device__ int d_queue[MAX_QUEUE_SIZE];
__device__ int d_head = 0;
__device__ int d_tail = 0;

// ==============================
// Инициализация очереди
// ==============================
__global__ void init_queue_kernel(int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_queue[idx] = idx; // детерминированные значения
    }
    if (idx == 0) {
        d_head = 0;
        d_tail = n;
    }
}

// ==============================
// Enqueue (добавление)
// ==============================
__global__ void enqueue_kernel(int* values, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int pos = atomicAdd(&d_tail, 1);
    if (pos < MAX_QUEUE_SIZE) {
        d_queue[pos] = values[idx];
    } else {
        atomicSub(&d_tail, 1); // rollback
    }
}

// ==============================
// Dequeue (удаление)
// ==============================
__global__ void dequeue_kernel(int* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int pos = atomicAdd(&d_head, 1);
    if (pos < d_tail) {
        output[idx] = d_queue[pos];
    } else {
        atomicSub(&d_head, 1); // rollback
        output[idx] = -1;
    }
}

// ==============================
// Печать очереди (CPU)
// ==============================
void print_queue(const vector<int>& data, int head, int tail) {
    int size = tail - head;
    cout << "Queue size = " << size << "\n";

    cout << "First 10: ";
    for (int i = head; i < min(head + 10, tail); i++)
        cout << data[i] << " ";
    cout << "\n";

    cout << "Last 10: ";
    for (int i = max(head, tail - 10); i < tail; i++)
        cout << data[i] << " ";
    cout << "\n\n";
}

// ==============================
// MAIN
// ==============================
int main() {
    cout << "=== Parallel Queue (FIFO) on GPU ===\n";

    const int N = 10000;

    // Данные для enqueue
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
    dim3 grid_init((INITIAL_QUEUE_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // ==============================
    // ИНИЦИАЛИЗАЦИЯ
    // ==============================
    init_queue_kernel<<<grid_init, block>>>(INITIAL_QUEUE_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    vector<int> h_queue(MAX_QUEUE_SIZE);
    int h_head, h_tail;

    CUDA_CHECK(cudaMemcpyFromSymbol(&h_head, d_head, sizeof(int)));
    CUDA_CHECK(cudaMemcpyFromSymbol(&h_tail, d_tail, sizeof(int)));
    CUDA_CHECK(cudaMemcpyFromSymbol(h_queue.data(), d_queue,
                                   MAX_QUEUE_SIZE * sizeof(int)));

    cout << "--- Before enqueue ---\n";
    print_queue(h_queue, h_head, h_tail);

    // ==============================
    // ENQUEUE
    // ==============================
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    enqueue_kernel<<<grid, block>>>(d_values, N);
    cudaEventRecord(stop);
    CUDA_CHECK(cudaDeviceSynchronize());

    float enqueue_time;
    cudaEventElapsedTime(&enqueue_time, start, stop);

    CUDA_CHECK(cudaMemcpyFromSymbol(&h_head, d_head, sizeof(int)));
    CUDA_CHECK(cudaMemcpyFromSymbol(&h_tail, d_tail, sizeof(int)));
    CUDA_CHECK(cudaMemcpyFromSymbol(h_queue.data(), d_queue,
                                   MAX_QUEUE_SIZE * sizeof(int)));

    cout << "--- After enqueue ---\n";
    print_queue(h_queue, h_head, h_tail);

    cout << "Expected queue size: " << INITIAL_QUEUE_SIZE + N << "\n";
    cout << "Actual queue size:   " << (h_tail - h_head) << "\n\n";

    // ==============================
    // DEQUEUE
    // ==============================
    cudaEventRecord(start);
    dequeue_kernel<<<grid, block>>>(d_output, N);
    cudaEventRecord(stop);
    CUDA_CHECK(cudaDeviceSynchronize());

    float dequeue_time;
    cudaEventElapsedTime(&dequeue_time, start, stop);

    CUDA_CHECK(cudaMemcpyFromSymbol(&h_head, d_head, sizeof(int)));
    CUDA_CHECK(cudaMemcpyFromSymbol(&h_tail, d_tail, sizeof(int)));
    CUDA_CHECK(cudaMemcpyFromSymbol(h_queue.data(), d_queue,
                                   MAX_QUEUE_SIZE * sizeof(int)));

    cout << "--- After dequeue ---\n";
    print_queue(h_queue, h_head, h_tail);

    cout << "Enqueue time: " << enqueue_time << " ms\n";
    cout << "Dequeue time: " << dequeue_time << " ms\n";

    cudaFree(d_values);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
