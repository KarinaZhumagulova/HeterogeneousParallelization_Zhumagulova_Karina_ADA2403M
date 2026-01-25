%%writefile assignment4_task2.cu
  
#include <iostream>        // Стандартный ввод-вывод
#include <vector>          // Поддержка контейнера динамических массивов (векторов)
#include <chrono>          // Библиотека для точного измерения времени на CPU
#include <random>          // Генератор случайных чисел
#include <cuda_runtime.h>  // Основной API CUDA для работы с видеокартой

#define N 1000000           // Общее количество элементов в массиве
#define BLOCK_SIZE 256      // Количество потоков в одном CUDA-блоке
#define RAND_MIN_VAL 1      // Минимальное значение для генерации чисел
#define RAND_MAX_VAL 100    // Максимальное значение для генерации чисел

// ---------------- Kernel 1: Сканирование на уровне блоков ----------------
// Выполняет префиксную сумму внутри каждого блока и сохраняет общую сумму блока
__global__ void blockScanKernel(int* d_input, int* d_output, int* d_block_sums, int n) {
    // Выделяем разделяемую память (shared memory) для быстрой работы внутри блока
    __shared__ int temp[BLOCK_SIZE];

    int tid = threadIdx.x;                           // Индекс потока внутри блока
    int gid = blockIdx.x * blockDim.x + tid;         // Глобальный индекс потока

    // Копируем данные из глобальной памяти в быструю разделяемую память
    // Если глобальный индекс за пределами массива, заполняем нулем
    temp[tid] = (gid < n) ? d_input[gid] : 0;
    __syncthreads();                                 // Ждем, пока все потоки загрузят данные

    // Алгоритм инклюзивного сканирования Hillis–Steele
    // На каждой итерации шаг (offset) удваивается
    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        int val = 0;
        if (tid >= offset)                           // Если у потока есть сосед слева на расстоянии offset
            val = temp[tid - offset];                // Берем его значение
        __syncthreads();                             // Синхронизация перед записью, чтобы не затереть нужные данные
        temp[tid] += val;                            // Суммируем
        __syncthreads();                             // Синхронизация перед следующей итерацией
    }

    // Записываем результат внутриблочного сканирования в выходной массив
    if (gid < n)
        d_output[gid] = temp[tid];

    // Последний поток каждого блока сохраняет полную сумму блока в отдельный массив
    // Это нужно для того, чтобы потом скорректировать значения в следующих блоках
    if (tid == blockDim.x - 1)
        d_block_sums[blockIdx.x] = temp[tid];
}

// ---------------- Kernel 2: Добавление сумм блоков к элементам ----------------
// Берет результаты сканирования сумм блоков и прибавляет их к соответствующим блокам
__global__ void addBlockSumsKernel(int* d_output, int* d_block_sums_scan, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x; // Глобальный индекс потока
    
    // Первый блок (index 0) не нуждается в коррекции, также проверяем границы массива
    if (blockIdx.x == 0 || gid >= n) return;

    // Берем префиксную сумму всех предыдущих блоков
    int add_val = d_block_sums_scan[blockIdx.x - 1];
    // Прибавляем эту сумму к текущему элементу
    d_output[gid] += add_val;
}

int main() {
    // ---------------- ИНИЦИАЛИЗАЦИЯ МАССИВОВ ----------------
    // Хост-векторы для входных данных, результата CPU и результата GPU
    std::vector<int> h_input(N), h_output_cpu(N), h_output_gpu(N);

    // Настройка генератора случайных чисел
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(RAND_MIN_VAL, RAND_MAX_VAL);

    // Заполнение входного массива случайными числами
    for (int i = 0; i < N; i++)
        h_input[i] = dist(gen);

    // ---------------- ПРЕФИКСНАЯ СУММА НА CPU ----------------
    auto cpu_start = std::chrono::high_resolution_clock::now(); // Замер времени начала
    h_output_cpu[0] = h_input[0];
    for (int i = 1; i < N; i++)
        h_output_cpu[i] = h_output_cpu[i - 1] + h_input[i]; // Последовательный алгоритм
    auto cpu_end = std::chrono::high_resolution_clock::now();   // Замер времени окончания
    std::chrono::duration<double, std::milli> cpu_time = cpu_end - cpu_start;

    // ---------------- ПРЕФИКСНАЯ СУММА НА GPU ----------------
    int *d_input, *d_output, *d_block_sums, *d_block_sums_scan;
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE; // Вычисляем необходимое кол-во блоков

    // Выделение видеопамяти под входные, выходные данные и промежуточные суммы блоков
    cudaMalloc(&d_input, N * sizeof(int));
    cudaMalloc(&d_output, N * sizeof(int));
    cudaMalloc(&d_block_sums, numBlocks * sizeof(int));
    cudaMalloc(&d_block_sums_scan, numBlocks * sizeof(int));

    // Копирование входных данных с хоста (RAM) на девайс (VRAM)
    cudaMemcpy(d_input, h_input.data(), N * sizeof(int), cudaMemcpyHostToDevice);

    // Создание событий CUDA для точного замера времени выполнения ядер
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start); // Фиксация времени старта

    // ЭТАП 1: Запуск ядра для локального сканирования внутри каждого блока
    
    blockScanKernel<<<numBlocks, BLOCK_SIZE>>>(d_input, d_output, d_block_sums, N);

    // ЭТАП 2: Сканирование сумм блоков
    // В данном коде это делается на CPU, так как количество блоков значительно меньше N
    std::vector<int> h_block_sums(numBlocks);
    // Копируем частичные суммы блоков обратно на CPU
    cudaMemcpy(h_block_sums.data(), d_block_sums, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);

    std::vector<int> h_block_sums_scan(numBlocks);
    h_block_sums_scan[0] = h_block_sums[0];
    for (int i = 1; i < numBlocks; i++)
        h_block_sums_scan[i] = h_block_sums_scan[i - 1] + h_block_sums[i]; // Префиксная сумма самих блоков

    // Копируем отсканированные суммы блоков обратно на GPU
    cudaMemcpy(d_block_sums_scan, h_block_sums_scan.data(), numBlocks * sizeof(int), cudaMemcpyHostToDevice);

    // ЭТАП 3: Финальная коррекция
    // Добавляем накопленную сумму предыдущих блоков к элементам текущего блока
    addBlockSumsKernel<<<numBlocks, BLOCK_SIZE>>>(d_output, d_block_sums_scan, N);

    // Копируем финальный результат с GPU на CPU
    cudaMemcpy(h_output_gpu.data(), d_output, N * sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaEventRecord(stop);      // Фиксация времени окончания
    cudaEventSynchronize(stop); // Ожидание завершения всех операций на GPU

    float gpu_time = 0.0f;
    cudaEventElapsedTime(&gpu_time, start, stop); // Расчет затраченного времени в мс

    // ---------------- ВЫВОД РЕЗУЛЬТАТОВ ----------------
    int print_count = 10;
    std::cout << "\nInput array - first " << print_count << ": ";
    for (int i = 0; i < print_count; i++) std::cout << h_input[i] << " ";
    std::cout << "\nInput array - last " << print_count << ": ";
    for (int i = N - print_count; i < N; i++) std::cout << h_input[i] << " ";

    std::cout << "\n\nCPU prefix sum - first " << print_count << ": ";
    for (int i = 0; i < print_count; i++) std::cout << h_output_cpu[i] << " ";
    std::cout << "\nCPU prefix sum - last " << print_count << ": ";
    for (int i = N - print_count; i < N; i++) std::cout << h_output_cpu[i] << " ";

    std::cout << "\n\nGPU prefix sum - first " << print_count << ": ";
    for (int i = 0; i < print_count; i++) std::cout << h_output_gpu[i] << " ";
    std::cout << "\nGPU prefix sum - last " << print_count << ": ";
    for (int i = N - print_count; i < N; i++) std::cout << h_output_gpu[i] << " ";

    std::cout << "\n\nArray size: " << N << std::endl;
    std::cout << "CPU time (ms): " << cpu_time.count() << std::endl;
    std::cout << "GPU time (ms): " << gpu_time << std::endl;

    // Освобождение выделенной видеопамяти
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_block_sums);
    cudaFree(d_block_sums_scan);

    return 0;
}
