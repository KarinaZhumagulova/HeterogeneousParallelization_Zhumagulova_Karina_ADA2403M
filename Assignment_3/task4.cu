%%writefile assignment3_task4.cu
#include <cuda_runtime.h> // Библиотека среды выполнения CUDA для работы с GPU
#include <iostream>       // Стандартная библиотека ввода-вывода
#include <random>         // Библиотека для генерации случайных чисел
#include <vector>         // Контейнер вектор для хранения результатов
#include <iomanip>        // Библиотека для форматирования вывода (например, точность чисел)

#define N 1000000         // Общее количество элементов в массивах
#define RAND_MIN_VAL 1    // Минимальное значение случайного числа
#define RAND_MAX_VAL 10000 // Максимальное значение случайного числа

using namespace std;

// -----------------------------
// Макрос для проверки ошибок выполнения CUDA
// -----------------------------
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        cerr << "CUDA error: " << cudaGetErrorString(err) \
             << " at " << __FILE__ << ":" << __LINE__ << endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// -----------------------------
// Ядро (Kernel): параллельное сложение векторов
// -----------------------------
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    // Рассчитываем уникальный глобальный индекс потока
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Проверка, чтобы индекс не вышел за пределы массива
    if(idx < n) c[idx] = a[idx] + b[idx];
}

// -----------------------------
// Вспомогательная функция для вывода краев массива (начала и конца)
// -----------------------------
void print_edges(const float* a, const string& msg) {
    cout << msg << "\nFirst 5: ";
    for(int i=0;i<5;i++) cout<<a[i]<<" "; // Печать первых 5 элементов
    cout<<"\nLast 5: ";
    for(int i=N-5;i<N;i++) cout<<a[i]<<" "; // Печать последних 5 элементов
    cout<<"\n\n";
}

// -----------------------------
// Функция запуска теста для конкретного размера блока
// -----------------------------
float run_test(int blockSize, const float* h_a, const float* h_b, float* h_c,
               float* d_a, float* d_b, float* d_c) {
    // Вычисляем размер сетки (Grid) в зависимости от размера блока
    int gridSize = (N + blockSize - 1)/blockSize;

    // Переменные для событий (таймеров)
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Записываем начало события
    CUDA_CHECK(cudaEventRecord(start));
    // Запуск ядра на GPU
    vector_add<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    // Записываем окончание события
    CUDA_CHECK(cudaEventRecord(stop));

    // Проверка на наличие ошибок запуска
    CUDA_CHECK(cudaGetLastError());
    // Ожидание завершения работы GPU
    CUDA_CHECK(cudaEventSynchronize(stop));

    float time_ms;
    // Получаем время в миллисекундах между событиями
    CUDA_CHECK(cudaEventElapsedTime(&time_ms, start, stop));

    // Копируем полученный результат из GPU в CPU (хост)
    CUDA_CHECK(cudaMemcpy(h_c, d_c, N*sizeof(float), cudaMemcpyDeviceToHost));

    // Освобождаем ресурсы событий
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return time_ms; // Возвращаем время выполнения
}

// -----------------------------
// Основная логика программы
// -----------------------------
int main() {
    // Проверка наличия устройств CUDA
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if(deviceCount==0) { cerr<<"No CUDA GPU found\n"; return 1; }

    size_t size = N * sizeof(float); // Размер памяти для массива в байтах
    float *h_a, *h_b, *h_c;          // Указатели для оперативной памяти (Host)
    float *d_a, *d_b, *d_c;          // Указатели для видеопамяти (Device)

    // Инициализация генератора случайных чисел
    random_device rd; mt19937 gen(rd()); uniform_int_distribution<> dist(RAND_MIN_VAL, RAND_MAX_VAL);
    
    // Выделение памяти на CPU
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c = (float*)malloc(size);
    
    // Заполнение входных массивов случайными числами
    for(int i=0;i<N;i++) { h_a[i] = dist(gen); h_b[i] = dist(gen); }

    // Выделение памяти на GPU
    CUDA_CHECK(cudaMalloc(&d_a,size));
    CUDA_CHECK(cudaMalloc(&d_b,size));
    CUDA_CHECK(cudaMalloc(&d_c,size));

    // Копирование входных векторов из CPU в GPU
    CUDA_CHECK(cudaMemcpy(d_a,h_a,size,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b,h_b,size,cudaMemcpyHostToDevice));

    // Вывод исходных данных
    print_edges(h_a,"Array A");
    print_edges(h_b,"Array B");

    // ===============================
    // Тестирование различных размеров блоков
    // ===============================
    vector<int> blockSizes = {32, 64, 128, 256, 512, 1024}; // Список размеров для проверки
    vector<float> times(blockSizes.size());                 // Массив для хранения времени
    float minTime = 1e9;                                    // Начальное значение для поиска минимума
    int optimalBlock = 0;                                   // Переменная для хранения лучшего размера блока

    cout<<"===== PERFORMANCE RESULTS =====\n";
    for(size_t i=0;i<blockSizes.size();i++) {
        // Выполняем тест и сохраняем время
        times[i] = run_test(blockSizes[i],h_a,h_b,h_c,d_a,d_b,d_c);
        cout<<"Block size "<<blockSizes[i]<<": "<<times[i]<<" ms\n";
        
        // Поиск минимального времени
        if(times[i] < minTime) {
            minTime = times[i];
            optimalBlock = blockSizes[i];
        }
    }

    // Вывод контрольного результата вычислений
    print_edges(h_c,"\nResult array C = A + B");

    // ===============================
    // Расчет эффективности
    // ===============================
    cout<<"\n===== OPTIMAL CONFIGURATION =====\n";
    cout<<"Optimal block size: "<<optimalBlock<<"\n";
    cout<<"Minimum execution time: "<<minTime<<" ms\n";

    cout<<"\n===== SPEEDUP RELATIVE TO OPTIMAL BLOCK =====\n";
    cout << fixed << setprecision(2); // Установка точности до 2 знаков после запятой
    for(size_t i=0;i<blockSizes.size();i++) {
        // Расчет замедления/ускорения (насколько текущее время больше оптимального)
        float speedup = times[i]/minTime;
        cout<<"Block size "<<blockSizes[i]<<": "<<speedup<<"x\n";
    }

    // Очистка памяти
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    free(h_a); free(h_b); free(h_c);

    return 0;
}
