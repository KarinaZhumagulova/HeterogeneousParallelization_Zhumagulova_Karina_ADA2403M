%%writefile task_1_array_addition.cu 

#include <iostream>       // Стандартный ввод-вывод
#include <vector>         // Контейнер для хранения данных на CPU
#include <chrono>         // Библиотека для высокоточного измерения времени на CPU
#include <random>         // Генератор случайных чисел
#include <cuda_runtime.h> // Основные функции CUDA API
#include <cmath>          // Математические функции (для fabs)

// Размер блока потоков (количество потоков в одной группе на GPU)
#define BLOCK_SIZE 256 
// Границы для генерации случайных чисел
#define RAND_MIN_VAL -100000
#define RAND_MAX_VAL  100000

// ================= CUDA KERNEL (Ядро) =================
// __global__ указывает, что функция вызывается с CPU, а исполняется на GPU
__global__ void vector_add_gpu(const float* A, const float* B, float* C, int n) {
    // Вычисление уникального индекса элемента для каждого потока
    // blockIdx.x - номер блока, blockDim.x - количество потоков в блоке, threadIdx.x - номер потока внутри блока
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    // Проверка, чтобы индекс не вышел за пределы массива (если N не кратно BLOCK_SIZE)
    if (id < n) {
        C[id] = A[id] + B[id]; // Само сложение
    }
}

// ================= CPU VERSION =================
// Обычная последовательная реализация сложения векторов на центральном процессоре
void vector_add_cpu(const float* A, const float* B, float* C, int n) {
    for (int i = 0; i < n; ++i) {
        C[i] = A[i] + B[i];
    }
}

// ================= PRINT FUNCTION =================
// Функция для вывода первых и последних элементов массива для визуальной проверки
void print_edges(const std::vector<float>& A,
                 const std::vector<float>& B,
                 const std::vector<float>& C,
                 int N) {
    int count = std::min(10, N); // Выводим максимум 10 элементов

    std::cout << "First " << count << " elements:\n";
    for (int i = 0; i < count; ++i) {
        std::cout << "A[" << i << "]=" << A[i]
                  << "  B[" << i << "]=" << B[i]
                  << "  C[" << i << "]=" << C[i] << "\n";
    }

    if (N > 10) {
        std::cout << "Last " << count << " elements:\n";
        for (int i = N - count; i < N; ++i) {
            std::cout << "A[" << i << "]=" << A[i]
                      << "  B[" << i << "]=" << B[i]
                      << "  C[" << i << "]=" << C[i] << "\n";
        }
    }
}

// ================= MAIN =================
int main() {
    // Набор размеров векторов для тестирования (от 10 до 10 миллионов элементов)
    std::vector<int> sizes = {
        10, 100, 1000, 10000, 100000, 1000000, 10000000
    };

    // Настройка генератора случайных чисел
    std::random_device rd;  // Инициализатор
    std::mt19937 gen(rd()); // Генератор (Вихрь Мерсенна)
    std::uniform_int_distribution<> dist(RAND_MIN_VAL, RAND_MAX_VAL); // Равномерное распределение

    std::cout << "Size\tCPU(ms)\tGPU(ms)\tSpeedup\n";

    // Цикл по всем тестовым размерам векторов
    for (int N : sizes) {
        size_t bytes = N * sizeof(float); // Объем памяти в байтах

        // Создание векторов в оперативной памяти (Host)
        std::vector<float> A(N), B(N), C_cpu(N), C_gpu(N);

        // Заполнение исходных данных случайными числами
        for (int i = 0; i < N; ++i) {
            A[i] = static_cast<float>(dist(gen));
            B[i] = static_cast<float>(dist(gen));
        }

        // ================= CPU TIMING =================
        auto cpu_start = std::chrono::high_resolution_clock::now(); // Засекаем время старта на CPU
        vector_add_cpu(A.data(), B.data(), C_cpu.data(), N);        // Вычисление на CPU
        auto cpu_end = std::chrono::high_resolution_clock::now();   // Засекаем время конца на CPU
        double cpu_time =
            std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count(); // Разница в миллисекундах

        // ================= GPU MEMORY ALLOCATION =================
        float *d_A, *d_B, *d_C; // Указатели для памяти на видеокарте (Device)
        cudaMalloc(&d_A, bytes); // Выделение памяти на GPU для вектора A
        cudaMalloc(&d_B, bytes); // Выделение памяти на GPU для вектора B
        cudaMalloc(&d_C, bytes); // Выделение памяти на GPU для вектора результата C

        // Копирование данных из ОЗУ в видеопамять
        cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B.data(), bytes, cudaMemcpyHostToDevice);

        // Расчет количества блоков: (N / размер блока) с округлением вверх
        int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // ================= GPU TIMING =================
        cudaEvent_t start, stop; // Создание специальных событий CUDA для замера времени
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start); // Фиксация события старта
        // Запуск ядра: <<<количество_блоков, количество_потоков_в_блоке>>>
        vector_add_gpu<<<gridSize, BLOCK_SIZE>>>(d_A, d_B, d_C, N);

        // Проверка на наличие ошибок при запуске ядра
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA kernel error: "
                      << cudaGetErrorString(err) << std::endl;
        }

        // Ожидание завершения всех операций на GPU перед замером времени
        cudaDeviceSynchronize();

        cudaEventRecord(stop);        // Фиксация события завершения
        cudaEventSynchronize(stop);   // Ожидание фактической записи события stop

        float gpu_time = 0.0f;
        cudaEventElapsedTime(&gpu_time, start, stop); // Расчет времени между событиями (в мс)

        // Копирование результата вычислений обратно с GPU на CPU
        cudaMemcpy(C_gpu.data(), d_C, bytes, cudaMemcpyDeviceToHost);

        // ================= VALIDATION =================
        // Проверка, совпадают ли результаты CPU и GPU с небольшой погрешностью
        bool correct = true;
        for (int i = 0; i < N; ++i) {
            if (fabs(C_cpu[i] - C_gpu[i]) > 1e-5) {
                correct = false;
                break;
            }
        }

        // Вывод итоговой таблицы: размер, время CPU, время GPU и ускорение
        std::cout << N << "\t"
                  << cpu_time << "\t"
                  << gpu_time << "\t"
                  << (cpu_time / gpu_time)
                  << (correct ? "" : " (ERROR)") << "\n";

        // Предпросмотр данных для текущего размера N
        std::cout << "=== Array preview for N = " << N << " ===\n";
        print_edges(A, B, C_gpu, N);
        std::cout << "========================================\n\n";

        // ================= CLEANUP =================
        cudaFree(d_A);           // Освобождение видеопамяти
        cudaFree(d_B);
        cudaFree(d_C);
        cudaEventDestroy(start); // Удаление событий замера времени
        cudaEventDestroy(stop);
    }

    return 0; // Завершение программы
}
