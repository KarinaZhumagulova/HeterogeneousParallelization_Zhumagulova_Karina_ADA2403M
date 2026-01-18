%%writefile assignment3_task1.cu
#include <cuda_runtime.h> // Основной заголовочный файл CUDA для работы с рантаймом
#include <iostream>       // Стандартный ввод-вывод C++
#include <random>         // Библиотека для генерации случайных чисел
#include <cstdlib>        // Стандартная библиотека (для malloc, exit)

// Константы программы
#define N 1000000         // Размер массива (1 миллион элементов)
#define BLOCK_SIZE 256    // Количество потоков в одном блоке CUDA
#define RAND_MIN_VAL 1    // Минимальное значение для рандома
#define RAND_MAX_VAL 100000 // Максимальное значение для рандома

using namespace std;

// ===============================
// Макрос для проверки ошибок CUDA
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
// Ядра (Kernels)
// ===============================

// Ядро, использующее только глобальную память GPU
__global__ void multiply_global(float* data, float value, int n) {
    // Вычисляем уникальный индекс потока во всей сетке
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Проверка границ: работаем только если индекс меньше размера массива
    if (idx < n) {
        data[idx] *= value; // Прямое чтение и запись в глобальную память
    }
}

// Ядро, использующее разделяемую (shared) память
__global__ void multiply_shared(float* data, float value, int n) {
    // Объявляем массив в разделяемой памяти (быстрая память внутри блока)
    __shared__ float buf[BLOCK_SIZE];
    
    // Глобальный индекс элемента
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        // 1. Копируем данные из медленной глобальной памяти в быструю разделяемую
        buf[threadIdx.x] = data[idx];
        
        // Синхронизация: ждем, пока все потоки блока заполнят буфер
        __syncthreads();
        
        // 2. Выполняем операцию в разделяемой памяти
        buf[threadIdx.x] *= value;
        
        // Синхронизация: ждем завершения вычислений перед записью обратно
        __syncthreads();
        
        // 3. Копируем результат обратно в глобальную память
        data[idx] = buf[threadIdx.x];
    }
}

// ===============================
// Вспомогательная функция для печати краев массива
// ===============================
void print_edges(const float* a, const string& msg) {
    cout << msg << "\nFirst 5: ";
    for (int i = 0; i < 5; i++) cout << a[i] << " "; // Печать первых 5 элементов
    cout << "\nLast 5:  ";
    for (int i = N - 5; i < N; i++) cout << a[i] << " "; // Печать последних 5
    cout << "\n\n";
}

// ===============================
// Основная функция
// ===============================
int main() {
    // Проверка наличия доступных GPU
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        cerr << "ERROR: No CUDA GPU found. Enable GPU in Colab.\n";
        return 1;
    }

    float *h_data, *h_res, *d_data; // Указатели для хоста (CPU) и устройства (GPU)
    float multiplier = 2.5f;        // Число, на которое будем умножать
    size_t size = N * sizeof(float); // Общий размер выделяемой памяти в байтах

    // Инициализация генератора случайных чисел
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dist(RAND_MIN_VAL, RAND_MAX_VAL);

    // Выделение памяти на хосте (RAM)
    h_data = (float*)malloc(size);
    h_res  = (float*)malloc(size);

    // Заполнение исходного массива случайными числами
    for (int i = 0; i < N; i++)
        h_data[i] = static_cast<float>(dist(gen));

    // Выделение памяти на устройстве (VRAM видеокарты)
    CUDA_CHECK(cudaMalloc(&d_data, size));

    // Расчет параметров сетки: сколько блоков нужно для N элементов
    int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Создание событий CUDA для замера времени выполнения
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float time_global = 0.0f;
    float time_shared = 0.0f;

    // ===============================
    // ТЕСТ 1: ГЛОБАЛЬНАЯ ПАМЯТЬ
    // ===============================
    cout << "=== GLOBAL MEMORY VERSION ===\n";
    print_edges(h_data, "Before");

    // Копируем данные с CPU на GPU
    CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(start)); // Старт таймера
    multiply_global<<<grid, BLOCK_SIZE>>>(d_data, multiplier, N); // Запуск ядра
    CUDA_CHECK(cudaEventRecord(stop));  // Стоп таймера

    CUDA_CHECK(cudaGetLastError());     // Проверка на ошибки при запуске ядра
    CUDA_CHECK(cudaEventSynchronize(stop)); // Ждем завершения работы GPU
    CUDA_CHECK(cudaEventElapsedTime(&time_global, start, stop)); // Считаем время

    // Копируем результат обратно на CPU для проверки
    CUDA_CHECK(cudaMemcpy(h_res, d_data, size, cudaMemcpyDeviceToHost));
    print_edges(h_res, "After");

    cout << "Global memory kernel time: " << time_global << " ms\n\n";

    // ===============================
    // ТЕСТ 2: РАЗДЕЛЯЕМАЯ ПАМЯТЬ
    // ===============================
    cout << "=== SHARED MEMORY VERSION ===\n";
    print_edges(h_data, "Before");

    // Снова копируем исходные данные на GPU
    CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(start)); // Старт таймера
    multiply_shared<<<grid, BLOCK_SIZE>>>(d_data, multiplier, N); // Запуск ядра
    CUDA_CHECK(cudaEventRecord(stop));  // Стоп таймера

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&time_shared, start, stop));

    CUDA_CHECK(cudaMemcpy(h_res, d_data, size, cudaMemcpyDeviceToHost));
    print_edges(h_res, "After");

    cout << "Shared memory kernel time: " << time_shared << " ms\n\n";

    // ===============================
    // ИТОГОВЫЙ ОТЧЕТ
    // ===============================
    cout << "===== PERFORMANCE SUMMARY =====\n";
    cout << "Array size: " << N << " elements\n";
    cout << "Global memory time: " << time_global << " ms\n";
    cout << "Shared memory time: " << time_shared << " ms\n";
    // Считаем коэффициент ускорения
    cout << "Speedup (global / shared): " << time_global / time_shared << "x\n";

    // Освобождение ресурсов
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data); // Очистка памяти видеокарты
    free(h_data);     // Очистка оперативной памяти
    free(h_res);

    return 0;
}
