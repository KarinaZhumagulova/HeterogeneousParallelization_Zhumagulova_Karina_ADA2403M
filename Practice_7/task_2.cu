%%writefile practice7_task2.cu
#include <iostream>        // Подключение стандартной библиотеки ввода-вывода
#include <vector>          // Подключение контейнера vector
#include <numeric>         // Подключение числовых алгоритмов
#include <random>          // Подключение генератора случайных чисел
#include <cuda_runtime.h>  // Подключение основного API CUDA
#include <chrono>          // Подключение функций для замера времени на CPU
#include <iomanip>         // Подключение манипуляторов форматирования вывода

using namespace std;

#define BLOCK_SIZE 256     // Определение количества потоков в одном блоке
#define RAND_MIN_VAL 0     // Минимальное значение для генерации
#define RAND_MAX_VAL 25    // Максимальное значение для генерации

// ---------------- GPU Ядро: Префиксная сумма через глобальную память (наивный подход) ----------------
__global__ void scanGlobalKernel(const float* d_input, float* d_output, int n) {
    // Вычисление глобального индекса потока
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Проверка на выход за границы массива
    if (gid < n) {
        float sum = 0.0f;
        // Каждый поток в цикле суммирует все элементы от начала до своего индекса
        // Это крайне неэффективно (сложность O(n^2) суммарно), используется для демонстрации
        for (int i = 0; i <= gid; ++i)
            sum += d_input[i];
        d_output[gid] = sum; // Запись результата в глобальную память
    }
}

// ---------------- GPU Ядро: Blelloch Scan с использованием разделяемой (shared) памяти ----------------
__global__ void scanSharedKernel(const float* d_input,
                                 float* d_output,
                                 float* d_blockSums,
                                 int n) {
    // Объявление массива в разделяемой памяти блока
    __shared__ float temp[BLOCK_SIZE];
    int tid = threadIdx.x;        // Локальный индекс потока в блоке
    int gid = blockIdx.x * blockDim.x + tid; // Глобальный индекс

    // Копирование данных из глобальной памяти в быструю разделяемую память
    temp[tid] = (gid < n) ? d_input[gid] : 0.0f;
    __syncthreads(); // Ожидание завершения копирования всеми потоками блока

    // Фаза Up-sweep (Reduce): построение сбалансированного бинарного дерева сумм
    for (int offset = 1; offset < BLOCK_SIZE; offset <<= 1) {
        int idx = (tid + 1) * offset * 2 - 1; // Вычисление индекса активного потока
        if (idx < BLOCK_SIZE)
            temp[idx] += temp[idx - offset];
        __syncthreads(); // Синхронизация на каждом уровне дерева
    }

    // Сохранение полной суммы блока для последующей коррекции других блоков
    if (tid == 0) {
        d_blockSums[blockIdx.x] = temp[BLOCK_SIZE - 1];
        temp[BLOCK_SIZE - 1] = 0.0f; // Обнуление последнего элемента для фазы Down-sweep
    }
    __syncthreads();

    // Фаза Down-sweep: распределение накопленных сумм обратно по дереву
    for (int offset = BLOCK_SIZE >> 1; offset > 0; offset >>= 1) {
        int idx = (tid + 1) * offset * 2 - 1;
        if (idx < BLOCK_SIZE) {
            float t = temp[idx - offset];
            temp[idx - offset] = temp[idx];      // Левому потомку передаем значение текущего узла
            temp[idx] += t;                     // Правому — сумму текущего и старого левого
        }
        __syncthreads();
    }

    // Запись инклюзивного результата (Blelloch дает эксклюзивный, добавляем сам элемент)
    if (gid < n)
        d_output[gid] = temp[tid] + d_input[gid];
}

// Ядро для добавления смещений блоков (чтобы объединить локальные сканы в глобальный)
__global__ void addOffsetsKernel(float* d_data,
                                 const float* d_blockOffsets,
                                 int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    // Ко всем элементам (кроме первого блока) прибавляем сумму всех предыдущих блоков
    if (gid < n && blockIdx.x > 0) {
        d_data[gid] += d_blockOffsets[blockIdx.x];
    }
}

// ---------------- Префиксная сумма на CPU (эталон) ----------------
void cpuPrefixSum(const vector<float>& input, vector<float>& output) {
    output[0] = input[0];
    for (size_t i = 1; i < input.size(); ++i)
        output[i] = output[i - 1] + input[i];
}

// ---------------- Обертка для запуска GPU Scan через глобальную память ----------------
float gpuGlobalPrefixSum(const vector<float>& h_input, vector<float>& h_output, int N) {
    float *d_input, *d_output;
    size_t bytes = N * sizeof(float);

    // Выделение памяти на видеокарте
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    // Копирование входных данных на GPU
    cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice);

    // Создание событий для замера времени
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Расчет количества блоков и запуск ядра
    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    scanGlobalKernel<<<blocks, BLOCK_SIZE>>>(d_input, d_output, N);
    
    cudaDeviceSynchronize(); // Ожидание завершения работы GPU
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop); // Получение времени выполнения

    // Копирование результата обратно на хост
    cudaMemcpy(h_output.data(), d_output, bytes, cudaMemcpyDeviceToHost);
    
    // Освобождение ресурсов
    cudaFree(d_input);
    cudaFree(d_output);

    return gpu_time;
}

// ---------------- Обертка для запуска GPU Scan через разделяемую память ----------------
float gpuSharedPrefixSum(const vector<float>& h_input, vector<float>& h_output, int N) {
    float *d_input, *d_output, *d_blockSums, *d_blockOffsets;
    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t bytes = N * sizeof(float);

    // Выделение памяти
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    cudaMalloc(&d_blockSums, blocks * sizeof(float));    // Для сумм каждого блока
    cudaMalloc(&d_blockOffsets, blocks * sizeof(float)); // Для накопленных смещений
    cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Шаг 1: Параллельное сканирование внутри каждого блока
    scanSharedKernel<<<blocks, BLOCK_SIZE>>>(d_input, d_output, d_blockSums, N);
    cudaDeviceSynchronize();

    // Шаг 2: Сканирование сумм блоков на стороне CPU (для простоты реализации)
    vector<float> h_blockSums(blocks);
    vector<float> h_blockOffsets(blocks, 0.0f);
    cudaMemcpy(h_blockSums.data(), d_blockSums, blocks * sizeof(float), cudaMemcpyDeviceToHost);

    // Вычисляем префиксную сумму самих блоков
    for (int i = 1; i < blocks; ++i)
        h_blockOffsets[i] = h_blockOffsets[i - 1] + h_blockSums[i - 1];

    // Копируем смещения обратно на GPU
    cudaMemcpy(d_blockOffsets, h_blockOffsets.data(), blocks * sizeof(float), cudaMemcpyHostToDevice);

    // Шаг 3: Добавляем смещения к результатам каждого блока на GPU
    addOffsetsKernel<<<blocks, BLOCK_SIZE>>>(d_output, d_blockOffsets, N);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop);

    cudaMemcpy(h_output.data(), d_output, bytes, cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_blockSums);
    cudaFree(d_blockOffsets);

    return gpu_time;
}

// ---------------- Основная функция ----------------
int main() {
    // Массив размеров входных данных для тестирования
    int sizes[] = {10, 100, 1000, 10000, 100000, 1000000};
    
    // Инициализация генератора случайных чисел
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dist(RAND_MIN_VAL, RAND_MAX_VAL);

    // Цикл по всем размерам тестов
    for (int s = 0; s < 6; ++s) {
        int N = sizes[s];

        // Подготовка векторов на CPU
        vector<float> h_input(N);
        vector<float> h_cpu(N);
        vector<float> h_gpu_global(N);
        vector<float> h_gpu_shared(N);

        // Заполнение входного массива случайными числами
        for (int i = 0; i < N; ++i)
            h_input[i] = static_cast<float>(dist(gen));

        // Выполнение на CPU и замер времени
        auto cpu_start = chrono::high_resolution_clock::now();
        cpuPrefixSum(h_input, h_cpu);
        auto cpu_end = chrono::high_resolution_clock::now();
        double cpu_time = chrono::duration<double, milli>(cpu_end - cpu_start).count();

        // Выполнение на GPU (глобальная память)
        float gpu_global_time = gpuGlobalPrefixSum(h_input, h_gpu_global, N);

        // Выполнение на GPU (разделяемая память)
        float gpu_shared_time = gpuSharedPrefixSum(h_input, h_gpu_shared, N);

        // Проверка корректности вычислений (сравнение с CPU)
        bool correct_global = true, correct_shared = true;
        for (int i = 0; i < N; ++i) {
            if (abs(h_cpu[i] - h_gpu_global[i]) > 1e-3) correct_global = false;
            if (abs(h_cpu[i] - h_gpu_shared[i]) > 1e-3) correct_shared = false;
        }

        // Вывод результатов в консоль
        cout << fixed << setprecision(6);
        cout << "N = " << N << endl;
        cout << "CPU time: " << cpu_time << " ms" << endl;
        cout << "GPU time (global memory): " << gpu_global_time << " ms" << endl;
        cout << "GPU time (shared memory): " << gpu_shared_time << " ms" << endl;
        cout << "Check: Global=" << (correct_global ? "CORRECT" : "INCORRECT")
             << ", Shared=" << (correct_shared ? "CORRECT" : "INCORRECT") << endl;

        // Вывод последних элементов для визуального подтверждения
        cout << "CPU last element: " << h_cpu[N-1] << endl;
        cout << "GPU (global memory) last element: " << h_gpu_global[N-1] << endl;
        cout << "GPU (shared memory) last element: " << h_gpu_shared[N-1] << endl;

        cout << "-----------------------------\n";
    }

    return 0; // Завершение программы
}
