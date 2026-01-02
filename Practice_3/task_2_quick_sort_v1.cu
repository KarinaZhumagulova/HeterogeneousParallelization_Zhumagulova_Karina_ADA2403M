%%writefile task_2_quick_sort_v1.cu
#include <iostream>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <cuda_runtime.h>
#include <iomanip>
#include <random>
#include <chrono>

using namespace std;

// =======================================================
// CUDA kernel: iterative quick sort per thread
// =======================================================
__global__ void kernel_quicksort(float* data, int n, int items_per_thread) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int left = tid * items_per_thread;
    int right = left + items_per_thread - 1;

    if (left < n) {
        if (right >= n) right = n - 1;

        int stack[64];
        int top = -1;
        stack[++top] = left;
        stack[++top] = right;

        while (top >= 0) {
            int r = stack[top--];
            int l = stack[top--];
            if (l >= r) continue;

            float pivot = data[(l + r) / 2];
            int i = l, j = r;

            while (i <= j) {
                while (data[i] < pivot) i++;
                while (data[j] > pivot) j--;
                if (i <= j) {
                    float tmp = data[i];
                    data[i] = data[j];
                    data[j] = tmp;
                    i++; j--;
                }
            }
            if (l < j) { stack[++top] = l; stack[++top] = j; }
            if (i < r) { stack[++top] = i; stack[++top] = r; }
        }
    }
}

// =======================================================
// Utility: print first & last 10 elements
// =======================================================
void print_edges(const vector<float>& arr) {
    int n = arr.size();
    int k = min(10, n);

    cout << "Первые 10: ";
    for (int i = 0; i < k; ++i)
        cout << (int)arr[i] << " ";
    cout << endl;

    cout << "Последние 10: ";
    for (int i = n - k; i < n; ++i)
        cout << (int)arr[i] << " ";
    cout << endl;
}

bool is_sorted_array(const vector<float>& arr) {
    for (size_t i = 1; i < arr.size(); ++i)
        if (arr[i] < arr[i - 1])
            return false;
    return true;
}

// =======================================================
// Test runner
// =======================================================
void run_test(int n) {
    vector<float> h_data(n);

    // === Correct random generation ===
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dist(-100000, 100000);

    #pragma omp parallel for
    for (int i = 0; i < n; ++i)
        h_data[i] = dist(gen);

    cout << "\n====================================\n";
    cout << "Размер массива: " << n << endl;

    cout << "До сортировки:\n";
    print_edges(h_data);

    float* d_data;
    cudaMalloc(&d_data, n * sizeof(float));
    cudaMemcpy(d_data, h_data.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 1024;
    int items_per_thread = (n + threads - 1) / threads;

    // === Time measurement with chrono ===
    auto start = chrono::high_resolution_clock::now();

    kernel_quicksort<<<1, threads>>>(d_data, n, items_per_thread);
    cudaDeviceSynchronize();

    cudaMemcpy(h_data.data(), d_data, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Final global sort
    sort(h_data.begin(), h_data.end());

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> elapsed = end - start;

    cudaFree(d_data);

    cout << "После сортировки:\n";
    print_edges(h_data);

    cout << "Время выполнения: " << fixed << setprecision(3)
         << elapsed.count() << " мс" << endl;

    cout << "Массив отсортирован: "
         << (is_sorted_array(h_data) ? "ДА" : "НЕТ") << endl;
}

// =======================================================
// Main
// =======================================================
int main() {
    vector<int> sizes = {100, 1000, 10000, 100000, 1000000};
    for (int n : sizes)
        run_test(n);
    return 0;
}
