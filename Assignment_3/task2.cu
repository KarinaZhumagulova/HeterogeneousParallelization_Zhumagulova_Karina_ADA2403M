%%writefile assignment3_task2.cu
#include <cuda_runtime.h> // Заголовочный файл для функций управления памятью и устройствами CUDA
#include <iostream>       // Стандартный поток ввода-вывода C++
#include <random>         // Современные средства C++ для генерации случайных чисел
#include <cstdlib>        // Стандартные функции C (malloc, free, exit)

#define N 1000000         // Длина векторов (1 миллион элементов)
#define RAND_MIN_VAL 1    // Нижняя граница случайных чисел
#define RAND_MAX_VAL 100  // Верхняя граница случайных чисел

using namespace std;

// ===============================
// Макрос для проверки ошибок выполнения функций CUDA
// ===============================
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        cerr << "CUDA error: " << cudaGetErrorString(err) \
             << " at " << __FILE__ << ":" << __LINE__ << endl; \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// ===============================
// Ядро (Kernel): сложение векторов
// ===============================
// __global__ указывает, что функция вызывается с CPU и исполняется на GPU
__global__ void vector_add(const float* a,
                           const float* b,
                           float* c,
                           int n) {
    // Вычисляем глобальный индекс потока: индекс блока * размер блока + индекс потока в блоке
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Проверка границ, чтобы потоки не вышли за пределы массива N
    if (idx < n) {
        c[idx] = a[idx] + b[idx]; // Каждый поток складывает один элемент
    }
}

// ===============================
// Вспомогательная функция для вывода первых и последних 5 элементов массива
// ===============================
void print_edges(const float* a, const string& msg) {
    cout << msg << "\nFirst 5: ";
    for (int i = 0; i < 5; i++) cout << a[i] << " "; // Начало массива
    cout << "\nLast 5:  ";
    for (int i = N - 5; i < N; i++) cout << a[i] << " "; // Конец массива
    cout << "\n\n";
}

// ===============================
// Функция запуска теста для конкретного размера блока
// ===============================
float run_test(int blockSize,
               const float* h_a,
               const float* h_b,
               float* h_c,
               float* d_a,
               float* d_b,
               float* d_c) {

    // Рассчитываем количество блоков в сетке (Grid), чтобы покрыть N элементов
    int gridSize = (N + blockSize - 1) / blockSize;

    // Создаем события CUDA для точного измерения времени работы GPU
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Записываем событие начала
    CUDA_CHECK(cudaEventRecord(start));
    
    // Запуск ядра с заданными параметрами сетки и блока
    vector_add<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    
    // Записываем событие окончания
    CUDA_CHECK(cudaEventRecord(stop));

    // Проверяем наличие ошибок при запуске ядра
    CUDA_CHECK(cudaGetLastError());
    
    // Ждем, пока GPU закончит выполнение всех задач до события stop
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Вычисляем разницу во времени между событиями
    float time_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&time_ms, start, stop));

    // Копируем результат сложения с GPU обратно на CPU для возможной проверки
    CUDA_CHECK(cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Удаляем события для освобождения ресурсов
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return time_ms; // Возвращаем время выполнения в миллисекундах
}

// ===============================
// Главная функция программы
// ===============================
int main() {
    // Проверяем наличие GPU с поддержкой CUDA
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        cerr << "ERROR: No CUDA GPU found.\n";
        return 1;
    }

    size_t size = N * sizeof(float); // Объем памяти для массивов

    // Указатели для памяти на хосте (CPU) и устройстве (GPU)
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;

    // Инициализация генератора случайных чисел
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dist(RAND_MIN_VAL, RAND_MAX_VAL);

    // Выделение оперативной памяти на CPU
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c = (float*)malloc(size);

    // Заполнение входных массивов случайными числами
    for (int i = 0; i < N; i++) {
        h_a[i] = static_cast<float>(dist(gen));
        h_b[i] = static_cast<float>(dist(gen));
    }

    // Выделение видеопамяти на GPU
    CUDA_CHECK(cudaMalloc(&d_a, size));
    CUDA_CHECK(cudaMalloc(&d_b, size));
    CUDA_CHECK(cudaMalloc(&d_c, size));

    // Копирование исходных данных из RAM (CPU) в VRAM (GPU)
    CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    // Вывод исходных значений (краев)
    print_edges(h_a, "Array A");
    print_edges(h_b, "Array B");

    // Список размеров блоков, которые мы хотим протестировать
    int blockSizes[] = {32, 64, 128, 256, 512};

    cout << "===== PERFORMANCE RESULTS =====\n";
    cout << "Array size: " << N << " elements\n\n";

    // Цикл по разным размерам блока для сравнения производительности
    for (int bs : blockSizes) {
        float time = run_test(bs, h_a, h_b, h_c, d_a, d_b, d_c);
        cout << "Block size " << bs
             << ": " << time << " ms\n";
    }

    // Вывод итогового массива C (краев) для подтверждения корректности
    print_edges(h_c, "\nResult array C = A + B");

    // Освобождение памяти на видеокарте
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    // Освобождение памяти в системе
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
