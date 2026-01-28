%%writefile practice7_task1.cu
#include <iostream>        // Подключение стандартной библиотеки ввода-вывода
#include <vector>          // Подключение контейнера vector
#include <numeric>         // Подключение числовых алгоритмов
#include <random>          // Подключение генератора случайных чисел
#include <cuda_runtime.h>  // Подключение основного API CUDA
#include <chrono>          // Подключение функций для замера времени на CPU
#include <iomanip>         // Подключение манипуляторов форматирования вывода

using namespace std;

// Размер блока потоков (количество потоков в одном блоке)
#define BLOCK_SIZE 256
// Константы для генерации случайных чисел
#define RAND_MIN_VAL 0
#define RAND_MAX_VAL 25

// ---------------- GPU Kernel: глобальная память ----------------
// Этот кернел демонстрирует базовый алгоритм редукции (суммирования)
__global__ void reduceGlobalKernel(const float* d_input, float* d_partial, int n) {
    // Вычисляем глобальный индекс потока
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    // Объявляем разделяемую (shared) память для потоков внутри одного блока
    __shared__ float sdata[BLOCK_SIZE];

    // Загружаем данные из глобальной памяти в разделяемую
    // Если индекс выходит за пределы массива, записываем 0
    sdata[threadIdx.x] = (i < n) ? d_input[i] : 0.0f;
    // Синхронизируем потоки в блоке, чтобы все данные были загружены в sdata
    __syncthreads();

    // Цикл редукции: складываем элементы парами внутри разделяемой памяти
    // s >>= 1 — это деление на 2 на каждой итерации (древовидная схема)
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        // Только первая половина активных потоков выполняет сложение
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        // Ждем завершения операции всеми потоками блока перед следующим шагом
        __syncthreads();
    }

    // Поток 0 записывает результат суммирования всего блока в глобальную память
    if (threadIdx.x == 0) d_partial[blockIdx.x] = sdata[0];
}

// ---------------- GPU Kernel: shared memory ----------------
// По логике этот кернел идентичен первому, так как в первом уже используется shared memory.
// В CUDA "редукция в глобальной памяти" обычно означает отсутствие shared памяти вовсе, 
// но это крайне неэффективно.
__global__ void reduceSharedKernel(const float* d_input, float* d_partial, int n) {
    __shared__ float sdata[BLOCK_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;

    // Шаг 1: Копирование из глобальной в разделяемую память
    sdata[tid] = (i < n) ? d_input[i] : 0.0f;
    __syncthreads();

    // Шаг 2: Внутриблоковая редукция
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Шаг 3: Запись частичной суммы блока в выходной массив
    if (tid == 0) d_partial[blockIdx.x] = sdata[0];
}

// ---------------- CPU reference ----------------
// Эталонная функция суммирования на процессоре для проверки точности
float cpuReduce(const vector<float>& data) {
    // accumulate — стандартная функция C++ для суммирования элементов
    return accumulate(data.begin(), data.end(), 0.0f);
}

// ---------------- GPU wrapper (Глобальная память) ----------------
float gpuReduceGlobal(const vector<float>& h_input, float& gpu_result) {
    int N = h_input.size();
    // Вычисляем количество блоков, необходимых для обработки всех данных
    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t bytes = N * sizeof(float);

    float *d_input, *d_partial;
    // Выделяем память на видеокарте
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_partial, blocks * sizeof(float));
    // Копируем входные данные с хоста (CPU) на девайс (GPU)
    cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice);

    // Создаем события CUDA для измерения времени
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start); // Фиксируем время начала

    // Запуск кернела
    reduceGlobalKernel<<<blocks, BLOCK_SIZE>>>(d_input, d_partial, N);
    
    // Копируем частичные суммы блоков обратно на CPU
    vector<float> h_partial(blocks);
    cudaMemcpy(h_partial.data(), d_partial, blocks * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Финальное суммирование частичных результатов на CPU
    gpu_result = cpuReduce(h_partial);

    cudaEventRecord(stop); // Фиксируем время окончания
    cudaEventSynchronize(stop); // Ждем завершения записи события
    
    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop); // Вычисляем разницу в мс

    // Освобождаем ресурсы
    cudaFree(d_input);
    cudaFree(d_partial);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return gpu_time;
}

// ---------------- GPU wrapper (Shared memory) ----------------
float gpuReduceShared(const vector<float>& h_input, float& gpu_result) {
    int N = h_input.size();
    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t bytes = N * sizeof(float);

    float *d_input, *d_partial;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_partial, blocks * sizeof(float));
    cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Вызов кернела, использующего shared memory
    reduceSharedKernel<<<blocks, BLOCK_SIZE>>>(d_input, d_partial, N);

    vector<float> h_partial(blocks);
    cudaMemcpy(h_partial.data(), d_partial, blocks * sizeof(float), cudaMemcpyDeviceToHost);
    gpu_result = cpuReduce(h_partial);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop);

    cudaFree(d_input);
    cudaFree(d_partial);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return gpu_time;
}

// ---------------- Main ----------------
int main() {
    // Наборы размеров входных данных для тестирования
    int sizes[] = {10, 100, 1000, 10000, 100000, 1000000};
    
    // Инициализация генератора случайных чисел
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dist(RAND_MIN_VAL, RAND_MAX_VAL);

    for (int s = 0; s < 6; ++s) {
        int N = sizes[s];

        // Создаем и заполняем вектор случайными числами
        vector<float> h_input(N);
        for (int i = 0; i < N; ++i) h_input[i] = static_cast<float>(dist(gen));

        // --- Тест на CPU ---
        auto cpu_start = chrono::high_resolution_clock::now();
        float cpu_result = cpuReduce(h_input);
        auto cpu_end = chrono::high_resolution_clock::now();
        double cpu_time = chrono::duration<double, milli>(cpu_end - cpu_start).count();

        // --- Тест на GPU ---
        float gpu_result_global, gpu_result_shared;
        float gpu_time_global = gpuReduceGlobal(h_input, gpu_result_global);
        float gpu_time_shared = gpuReduceShared(h_input, gpu_result_shared);

        // Проверка корректности (с учетом погрешности float)
        bool correct_global = abs(cpu_result - gpu_result_global) < 1e-2;
        bool correct_shared = abs(cpu_result - gpu_result_shared) < 1e-2;

        // Вывод результатов в консоль
        cout << fixed << setprecision(6);
        cout << "N = " << N << endl;
        cout << "CPU time: " << cpu_time << " ms" << endl;
        cout << "GPU time (global memory): " << gpu_time_global << " ms" << endl;
        cout << "GPU time (shared memory): " << gpu_time_shared << " ms" << endl;
        cout << "Check: Global=" << (correct_global ? "CORRECT" : "INCORRECT")
             << ", Shared=" << (correct_shared ? "CORRECT" : "INCORRECT") << endl;
        cout << "CPU Sum: " << cpu_result << endl;
        cout << "GPU Global Sum: " << gpu_result_global << endl;
        cout << "GPU Shared Sum: " << gpu_result_shared << endl;
        cout << "-----------------------------\n";
    }

    return 0;
}
