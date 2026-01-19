%%writefile task_2_matrix_multiplication.cu

#include <iostream>    // Стандартный поток ввода-вывода
#include <vector>      // Контейнер векторов C++
#include <random>      // Библиотека для генерации случайных чисел
#include <cuda_runtime.h> // Основной заголовочный файл CUDA

// Размер квадратного блока потоков (16x16 = 256 потоков в блоке)
#define BLOCK_SIZE 16
// Диапазон случайных чисел для заполнения матриц
#define RAND_MIN_VAL -100
#define RAND_MAX_VAL  100

// ================= CUDA KERNEL (Ядро) =================
// Функция, выполняемая на GPU. Каждый поток вычисляет один элемент результирующей матрицы C.
__global__ void matrix_mul_gpu(
    const int* A, // Указатель на матрицу A в видеопамяти
    const int* B, // Указатель на матрицу B в видеопамяти
    int* C,       // Указатель на матрицу результата C в видеопамяти
    int N         // Размер стороны матрицы (NxN)
) {
    // Вычисляем индекс строки и столбца для текущего потока в двумерной сетке
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Проверка на выход за границы матрицы (важно, если N не кратно BLOCK_SIZE)
    if (row < N && col < N) {
        int sum = 0; // Временная переменная для накопления суммы (скалярное произведение)
        // Цикл по строке A и столбцу B
        for (int k = 0; k < N; ++k) {
            // Формула индексации в одномерном массиве для 2D данных: [строка * ширина + столбец]
            sum += A[row * N + k] * B[k * N + col];
        }
        // Запись итогового значения в ячейку матрицы C
        C[row * N + col] = sum;
    }
}

// ================= CPU VERSION =================
// Классическое умножение матриц с тремя вложенными циклами (O(N^3))
void matrix_mul_cpu(
    const int* A,
    const int* B,
    int* C,
    int N
) {
    for (int i = 0; i < N; ++i) // Проход по строкам
        for (int j = 0; j < N; ++j) { // Проход по столбцам
            int sum = 0;
            for (int k = 0; k < N; ++k) // Скалярное произведение строки на столбец
                sum += A[i * N + k] * B[k * N + j];
            C[i * N + j] = sum; // Сохранение результата
        }
}

// ================= PRINT MATRIX =================
// Вспомогательная функция для вывода матрицы в консоль в удобном виде
void print_matrix(const std::vector<int>& M, int N, const std::string& name) {
    std::cout << name << ":\n";
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j)
            std::cout << M[i * N + j] << "\t"; // Табуляция для выравнивания колонок
        std::cout << "\n";
    }
    std::cout << "\n";
}

// ================= MAIN =================
int main() {
    // Тестовые размеры маленьких матриц для отладки и проверки корректности
    std::vector<int> sizes = {2, 3, 4, 5, 10};

    // Инициализация генератора случайных чисел (фикс. зерно 42 для повторяемости)
    std::mt19937 gen(42);
    std::uniform_int_distribution<> dist(RAND_MIN_VAL, RAND_MAX_VAL);

    for (int N : sizes) {
        std::cout << "==============================\n";
        std::cout << "Matrix size: " << N << "x" << N << "\n";

        // Расчет объема памяти для матрицы NxN
        size_t bytes = N * N * sizeof(int);

        // Хост-векторы (память на CPU)
        std::vector<int> A(N * N), B(N * N), C_cpu(N * N), C_gpu(N * N);

        // Заполнение матриц A и B случайными числами
        for (auto& x : A) x = dist(gen);
        for (auto& x : B) x = dist(gen);

        // Вычисление на CPU для эталона
        matrix_mul_cpu(A.data(), B.data(), C_cpu.data(), N);

        // Подготовка памяти на GPU (Device)
        int *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, bytes); // Выделение памяти под A
        cudaMalloc(&d_B, bytes); // Выделение памяти под B
        cudaMalloc(&d_C, bytes); // Выделение памяти под результат C

        // Копирование входных данных с Host на Device
        cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B.data(), bytes, cudaMemcpyHostToDevice);

        // Определение конфигурации запуска ядра
        // dim3 - специальный тип CUDA для задания 2D/3D размеров
        dim3 block(BLOCK_SIZE, BLOCK_SIZE); // Блок 16x16 потоков
        dim3 grid(
            (N + BLOCK_SIZE - 1) / BLOCK_SIZE, // Количество блоков по горизонтали
            (N + BLOCK_SIZE - 1) / BLOCK_SIZE  // Количество блоков по вертикали
        );

        // Запуск ядра на исполнение
        matrix_mul_gpu<<<grid, block>>>(d_A, d_B, d_C, N);
        
        // Ожидание завершения работы GPU (синхронизация)
        cudaDeviceSynchronize();

        // Копирование результата обратно в память CPU
        cudaMemcpy(C_gpu.data(), d_C, bytes, cudaMemcpyDeviceToHost);

        // Вывод всех матриц для визуальной проверки
        print_matrix(A, N, "Matrix A");
        print_matrix(B, N, "Matrix B");
        print_matrix(C_cpu, N, "Matrix C (CPU)");
        print_matrix(C_gpu, N, "Matrix C (GPU)");

        // Валидация: сравнение каждого элемента CPU и GPU версий
        bool correct = true;
        for (int i = 0; i < N * N; ++i)
            if (C_cpu[i] != C_gpu[i]) {
                correct = false;
                break;
            }

        std::cout << "Correctness: "
                  << (correct ? "OK" : "ERROR") << "\n";

        // Освобождение выделенной видеопамяти
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }

    return 0;
}
