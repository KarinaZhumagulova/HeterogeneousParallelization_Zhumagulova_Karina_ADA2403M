%%writefile assignment4_task1.cu

#include <iostream>        // Стандартный ввод-вывод
#include <vector>          // Поддержка контейнера динамических массивов (векторов)
#include <chrono>          // Библиотека для точного измерения времени на CPU
#include <random>          // Генератор случайных чисел
#include <cuda_runtime.h>  // Основной API CUDA для работы с видеокартой

#define N 100000           // Размер массива (100 тысяч элементов)
#define BLOCK_SIZE 256     // Количество потоков в одном блоке CUDA
#define RAND_MIN_VAL -10000 // Минимальное значение случайного числа
#define RAND_MAX_VAL 10000  // Максимальное значение случайного числа

// CUDA-ядро: функция, которая выполняется параллельно на множестве ядер GPU
__global__ void sumKernel(int* d_array, unsigned long long* d_result, int n) {
    // Вычисляем глобальный индекс потока
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Проверяем, не выходит ли индекс за пределы массива
    if (idx < n) {
        // Атомарное сложение: безопасно прибавляет значение элемента к общей сумме.
        // atomicAdd гарантирует, что разные потоки не перезапишут данные друг друга одновременно.
        atomicAdd(d_result, (unsigned long long)d_array[idx]);
    }
}

int main() {
    // ---------------- ИНИЦИАЛИЗАЦИЯ СЛУЧАЙНОГО МАССИВА ----------------
    std::vector<int> h_array(N); // Создаем массив на хосте (CPU) размера N

    std::random_device rd;  // Источник энтропии для генератора
    std::mt19937 gen(rd()); // Генератор "Вихрь Мерсенна"
    std::uniform_int_distribution<> dist(RAND_MIN_VAL, RAND_MAX_VAL); // Равномерное распределение чисел

    // Заполняем массив случайными числами
    for (int i = 0; i < N; i++)
        h_array[i] = dist(gen);

    // ----------- ВЫВОД ЧАСТИ МАССИВА ДЛЯ ПРОВЕРКИ -----------
    std::cout << "First 10: ";
    for (int i = 0; i < 10; i++) std::cout << h_array[i] << " ";
    std::cout << "\nLast 10: ";
    for (int i = N - 10; i < N; i++) std::cout << h_array[i] << " ";
    std::cout << "\n";

    // ---------------- СУММИРОВАНИЕ НА CPU ----------------
    long long cpu_sum = 0;
    auto cpu_start = std::chrono::high_resolution_clock::now(); // Засекаем время начала
    
    for (int i = 0; i < N; i++)
        cpu_sum += h_array[i]; // Обычный цикл суммирования
        
    auto cpu_end = std::chrono::high_resolution_clock::now();   // Засекаем время конца

    // Вычисляем длительность в миллисекундах
    std::chrono::duration<double, std::milli> cpu_time = cpu_end - cpu_start;

    // ---------------- СУММИРОВАНИЕ НА GPU ----------------
    int* d_array;                   // Указатель для массива на видеокарте
    unsigned long long* d_result;   // Указатель для результата на видеокарте
    unsigned long long gpu_sum = 0; // Переменная для хранения итоговой суммы на CPU

    // Выделяем память в Глобальной памяти видеокарты (VRAM)
    cudaMalloc(&d_array, N * sizeof(int));
    cudaMalloc(&d_result, sizeof(unsigned long long));

    // Копируем данные из оперативной памяти (Host) в видеопамять (Device)
    cudaMemcpy(d_array, h_array.data(), N * sizeof(int), cudaMemcpyHostToDevice);
    // Обнуляем переменную результата на видеокарте
    cudaMemset(d_result, 0, sizeof(unsigned long long));

    // Рассчитываем количество блоков: (N / размер_блока) с округлением вверх
    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Создаем события CUDA для измерения времени выполнения ядра
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start); // Запись события "старт"
    // Запуск ядра: <<<количество_блоков, потоков_в_блоке>>>
    sumKernel<<<blocks, BLOCK_SIZE>>>(d_array, d_result, N);
    cudaEventRecord(stop);  // Запись события "стоп"

    // Копируем результат вычислений обратно с видеокарты на CPU
    cudaMemcpy(&gpu_sum, d_result, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    
    // Ждем завершения всех операций на GPU перед расчетом времени
    cudaEventSynchronize(stop);

    float gpu_time = 0.0f;
    // Считаем разницу времени между событиями start и stop
    cudaEventElapsedTime(&gpu_time, start, stop);

    // ---------------- ВЫВОД РЕЗУЛЬТАТОВ ----------------
    std::cout << "\nArray size: " << N << std::endl;
    std::cout << "CPU sum: " << cpu_sum << std::endl;
    std::cout << "GPU sum: " << (long long)gpu_sum << std::endl;
    std::cout << "CPU time (ms): " << cpu_time.count() << std::endl;
    std::cout << "GPU time (ms): " << gpu_time << std::endl;

    // Освобождаем выделенную видеопамять
    cudaFree(d_array);
    cudaFree(d_result);

    return 0; // Завершение программы
}
