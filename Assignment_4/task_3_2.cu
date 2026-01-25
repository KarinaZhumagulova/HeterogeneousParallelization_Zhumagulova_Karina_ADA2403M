%%writefile assignment4_task3_hybrid.cu

#include <iostream>        // Стандартный ввод-вывод
#include <vector>          // Поддержка контейнера динамических массивов (векторов)
#include <chrono>          // Библиотека для точного измерения времени на CPU
#include <random>          // Генератор случайных чисел
#include <cuda_runtime.h>  // Основной API CUDA для работы с видеокартой
#include <thread>          // Библиотека для работы с потоками CPU (std::thread)

#define N 1000000           // Размер массива
#define BLOCK_SIZE 256      // Потоков на блок
#define RAND_MIN_VAL -1000
#define RAND_MAX_VAL 1000

// ---------------- GPU Kernel ----------------
// Ядро для параллельных вычислений на видеокарте
__global__ void processKernel(int* d_array, int n) {
    // Вычисляем глобальный индекс текущего потока
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    // Проверка выхода за границы массива
    if (gid < n) {
        d_array[gid] *= 2; // Умножаем элемент на 2
    }
}

// ---------------- CPU processing function ----------------
// Функция для обработки данных на CPU (будет запущена в отдельном потоке)
void processCPU(std::vector<int>& arr, int start, int end) {
    // Обработка заданного диапазона [start, end)
    for (int i = start; i < end; i++) {
        arr[i] *= 2;
    }
}

int main() {
    // ---------------- INIT ARRAY ----------------
    // Подготовка векторов для хранения исходных данных и результатов разных тестов
    std::vector<int> h_array(N), h_cpu(N), h_gpu(N), h_hybrid(N);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(RAND_MIN_VAL, RAND_MAX_VAL);

    // Инициализация исходного массива случайными числами
    for (int i = 0; i < N; i++) {
        h_array[i] = dist(gen);
    }

    // ---------------- CPU PROCESS ----------------
    h_cpu = h_array; // Копируем данные для чистого CPU-теста
    auto cpu_start = std::chrono::high_resolution_clock::now();
    processCPU(h_cpu, 0, N); // Обрабатываем весь массив на CPU последовательно
    auto cpu_end = std::chrono::high_resolution_clock::now();
    // Расчет времени выполнения на CPU
    std::chrono::duration<double, std::milli> cpu_time = cpu_end - cpu_start;

    // ---------------- GPU PROCESS ----------------
    h_gpu = h_array; // Копируем данные для чистого GPU-теста
    int* d_array;
    // Выделяем память на видеокарте и копируем туда весь массив
    cudaMalloc(&d_array, N * sizeof(int));
    cudaMemcpy(d_array, h_gpu.data(), N * sizeof(int), cudaMemcpyHostToDevice);

    // Считаем количество блоков для покрытия N элементов
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Создание и фиксация событий для замера времени GPU
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Запуск ядра для обработки всего массива на GPU
    processKernel<<<numBlocks, BLOCK_SIZE>>>(d_array, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop); // Ждем завершения расчетов

    // Копируем результат обратно в h_gpu
    cudaMemcpy(h_gpu.data(), d_array, N * sizeof(int), cudaMemcpyDeviceToHost);

    float gpu_time = 0.0f;
    cudaEventElapsedTime(&gpu_time, start, stop); // Получаем время работы GPU в мс
    cudaFree(d_array); // Освобождаем память на видеокарте

    // ---------------- PARALLEL HYBRID PROCESS ----------------
    
    h_hybrid = h_array; // Копируем данные для гибридного теста
    int split = N / 2;  // Точка разделения нагрузки пополам

    auto hybrid_start = std::chrono::high_resolution_clock::now();

    // ЗАПУСК ПОТОКА CPU: обрабатывает первую половину массива [0, split)
    // std::ref используется для передачи вектора по ссылке
    std::thread cpu_thread(processCPU, std::ref(h_hybrid), 0, split);

    // ЗАПУСК GPU В ОСНОВНОМ ПОТОКЕ: обрабатывает вторую половину [split, N)
    int* d_hybrid;
    // Выделяем память на GPU только под вторую половину
    cudaMalloc(&d_hybrid, (N - split) * sizeof(int));
    // Копируем вторую половину данных на GPU
    cudaMemcpy(d_hybrid, &h_hybrid[split], (N - split) * sizeof(int), cudaMemcpyHostToDevice);

    // Расчет количества блоков для половины массива
    int blocks_hybrid = ((N - split) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // Вызываем ядро для обработки половины данных на GPU
    processKernel<<<blocks_hybrid, BLOCK_SIZE>>>(d_hybrid, N - split);
    cudaDeviceSynchronize(); // Ожидаем завершения вычислений на видеокарте

    // Копируем результат GPU-части обратно в исходный вектор на хосте
    cudaMemcpy(&h_hybrid[split], d_hybrid, (N - split) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_hybrid); // Освобождаем память GPU

    // СИНХРОНИЗАЦИЯ: Ждем, пока поток CPU завершит свою работу над первой половиной
    cpu_thread.join();

    auto hybrid_end = std::chrono::high_resolution_clock::now();
    // Расчет общего времени гибридной обработки (параллельное исполнение)
    float hybrid_time = std::chrono::duration<double, std::milli>(hybrid_end - hybrid_start).count();

    // ---------------- CHECK RESULTS (Вывод результатов) ----------------
    int print_count = 10;
    std::cout << "Original array - first " << print_count << ": ";
    for (int i = 0; i < print_count; i++) std::cout << h_array[i] << " ";
    std::cout << "\nOriginal array - last " << print_count << ": ";
    for (int i = N - print_count; i < N; i++) std::cout << h_array[i] << " ";

    std::cout << "\n\nCPU processed - first " << print_count << ": ";
    for (int i = 0; i < print_count; i++) std::cout << h_cpu[i] << " ";
    std::cout << "\nCPU processed - last " << print_count << ": ";
    for (int i = N - print_count; i < N; i++) std::cout << h_cpu[i] << " ";

    std::cout << "\n\nGPU processed - first " << print_count << ": ";
    for (int i = 0; i < print_count; i++) std::cout << h_gpu[i] << " ";
    std::cout << "\nGPU processed - last " << print_count << ": ";
    for (int i = N - print_count; i < N; i++) std::cout << h_gpu[i] << " ";

    std::cout << "\n\nHybrid processed - first " << print_count << ": ";
    for (int i = 0; i < print_count; i++) std::cout << h_hybrid[i] << " ";
    std::cout << "\nHybrid processed - last " << print_count << ": ";
    for (int i = N - print_count; i < N; i++) std::cout << h_hybrid[i] << " ";

    // ---------------- TIME RESULTS (Вывод временных показателей) ----------------
    std::cout << "\n\nArray size: " << N << std::endl;
    std::cout << "CPU time (ms): " << cpu_time.count() << std::endl;
    std::cout << "GPU time (ms): " << gpu_time << std::endl;
    std::cout << "Hybrid time (ms): " << hybrid_time << std::endl;

    return 0;
}
