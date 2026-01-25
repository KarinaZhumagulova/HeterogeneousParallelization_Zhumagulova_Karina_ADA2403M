%%writefile assignment4_task3.cu

#include <iostream>        // Стандартный ввод-вывод
#include <vector>          // Поддержка контейнера динамических массивов (векторов)
#include <chrono>          // Библиотека для точного измерения времени на CPU
#include <random>          // Генератор случайных чисел
#include <cuda_runtime.h>  // Основной API CUDA для работы с видеокартой

#define N 1000000           // Общее количество элементов в массиве (1 миллион)
#define BLOCK_SIZE 256      // Количество потоков в одном CUDA-блоке
#define RAND_MIN_VAL -1000  // Нижняя граница для генерации случайных чисел
#define RAND_MAX_VAL 1000   // Верхняя граница для генерации случайных чисел

// ---------------- GPU Kernel ----------------
// Ядро CUDA: выполняется параллельно на видеокарте
__global__ void processKernel(int* d_array, int n) {
    // Вычисляем уникальный глобальный индекс потока
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Проверяем, не выходит ли индекс за пределы массива
    if (gid < n) {
        // Выполняем полезную работу: умножаем значение элемента на 2
        d_array[gid] *= 2;
    }
}

int main() {
    // ---------------- ИНИЦИАЛИЗАЦИЯ МАССИВОВ ----------------
    // Резервируем память на хосте (CPU) для исходного массива и результатов разных типов обработки
    std::vector<int> h_array(N), h_cpu(N), h_gpu(N), h_hybrid(N);

    // Подготовка генератора случайных чисел
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(RAND_MIN_VAL, RAND_MAX_VAL);

    // Заполнение исходного массива случайными данными
    for (int i = 0; i < N; i++) {
        h_array[i] = dist(gen);
    }

    // ---------------- ОБРАБОТКА НА CPU ----------------
    h_cpu = h_array; // Копируем данные в массив для обработки на процессоре
    auto cpu_start = std::chrono::high_resolution_clock::now(); // Засекаем время начала
    for (int i = 0; i < N; i++) {
        h_cpu[i] *= 2; // Последовательное умножение каждого элемента
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();   // Засекаем время окончания
    // Вычисляем длительность в миллисекундах
    std::chrono::duration<double, std::milli> cpu_time = cpu_end - cpu_start;

    // ---------------- ОБРАБОТКА НА GPU ----------------
    h_gpu = h_array; // Подготовка массива для GPU-теста
    int* d_array;    // Указатель для памяти на видеокарте
    
    // Выделяем память в VRAM и копируем туда данные с хоста
    cudaMalloc(&d_array, N * sizeof(int));
    cudaMemcpy(d_array, h_gpu.data(), N * sizeof(int), cudaMemcpyHostToDevice);

    // Расчет количества блоков, необходимых для покрытия всего массива N
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Настройка событий CUDA для точного замера времени работы видеокарты
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start); // Старт записи времени

    // Запуск вычислительного ядра на GPU
    processKernel<<<numBlocks, BLOCK_SIZE>>>(d_array, N);

    cudaEventRecord(stop);      // Остановка записи времени
    cudaEventSynchronize(stop); // Ждем завершения всех операций на GPU

    // Копируем обработанные данные обратно в оперативную память (RAM)
    cudaMemcpy(h_gpu.data(), d_array, N * sizeof(int), cudaMemcpyDeviceToHost);

    float gpu_time = 0.0f;
    cudaEventElapsedTime(&gpu_time, start, stop); // Получаем время работы в мс
    cudaFree(d_array); // Освобождаем память на видеокарте

    // ---------------- ГИБРИДНАЯ ОБРАБОТКА (CPU + GPU) ----------------
    int split = N / 2;  // Определяем точку разделения: половина на CPU, половина на GPU
    h_hybrid = h_array; // Копируем исходные данные

    // Фиксируем время начала всей гибридной операции
    auto hybrid_start = std::chrono::high_resolution_clock::now();
    
    // Часть 1: CPU обрабатывает первую половину массива
    for (int i = 0; i < split; i++) {
        h_hybrid[i] *= 2;
    }

    // Часть 2: GPU обрабатывает вторую половину массива
    int* d_hybrid;
    // Выделяем память только под оставшуюся половину (N - split)
    cudaMalloc(&d_hybrid, (N - split) * sizeof(int));
    // Копируем вторую часть данных на девайс, начиная со смещения split
    cudaMemcpy(d_hybrid, &h_hybrid[split], (N - split) * sizeof(int), cudaMemcpyHostToDevice);

    // Рассчитываем блоки для гибридной части
    int blocks_hybrid = ((N - split) + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Запуск ядра для обработки второй половины
    processKernel<<<blocks_hybrid, BLOCK_SIZE>>>(d_hybrid, N - split);
    cudaDeviceSynchronize(); // Синхронизация устройства
    
    // Копируем результат GPU-части обратно в массив h_hybrid по нужному адресу
    cudaMemcpy(&h_hybrid[split], d_hybrid, (N - split) * sizeof(int), cudaMemcpyDeviceToHost);
    auto hybrid_end = std::chrono::high_resolution_clock::now(); // Конец гибридной обработки

    // Рассчитываем общее время гибридного подхода (включая копирование и работу CPU)
    float hybrid_time = std::chrono::duration<double, std::milli>(hybrid_end - hybrid_start).count();
    cudaFree(d_hybrid); // Освобождаем выделенную память

    // ---------------- ПРОВЕРКА И ВЫВОД РЕЗУЛЬТАТОВ ----------------
    int print_count = 10; // Количество элементов для предпросмотра
    
    // Вывод первых и последних элементов исходного массива
    std::cout << "Original array - first " << print_count << ": ";
    for (int i = 0; i < print_count; i++) std::cout << h_array[i] << " ";
    std::cout << "\nOriginal array - last " << print_count << ": ";
    for (int i = N - print_count; i < N; i++) std::cout << h_array[i] << " ";

    // Вывод результатов CPU
    std::cout << "\n\nCPU processed - first " << print_count << ": ";
    for (int i = 0; i < print_count; i++) std::cout << h_cpu[i] << " ";
    std::cout << "\nCPU processed - last " << print_count << ": ";
    for (int i = N - print_count; i < N; i++) std::cout << h_cpu[i] << " ";

    // Вывод результатов GPU
    std::cout << "\n\nGPU processed - first " << print_count << ": ";
    for (int i = 0; i < print_count; i++) std::cout << h_gpu[i] << " ";
    std::cout << "\nGPU processed - last " << print_count << ": ";
    for (int i = N - print_count; i < N; i++) std::cout << h_gpu[i] << " ";

    // Вывод результатов гибридного метода
    std::cout << "\n\nHybrid processed - first " << print_count << ": ";
    for (int i = 0; i < print_count; i++) std::cout << h_hybrid[i] << " ";
    std::cout << "\nHybrid processed - last " << print_count << ": ";
    for (int i = N - print_count; i < N; i++) std::cout << h_hybrid[i] << " ";

    // ---------------- СРАВНЕНИЕ ВРЕМЕНИ ----------------
    std::cout << "\n\nArray size: " << N << std::endl;
    std::cout << "CPU time (ms): " << cpu_time.count() << std::endl;
    std::cout << "GPU time (ms): " << gpu_time << std::endl;
    std::cout << "Hybrid time (ms): " << hybrid_time << std::endl;

    return 0; // Завершение программы
}
