#include <cuda_runtime.h> // Библиотека среды исполнения CUDA
#include <iostream>       // Стандартный ввод-вывод
#include <vector>         // Контейнер vector
#include <random>         // Генерация случайных чисел

using namespace std;

// Константы для диапазона случайных чисел
#define RAND_MIN_VAL -100
#define RAND_MAX_VAL 100

// ---------------------------
// Печать первых 10 элементов массива для контроля данных
// ---------------------------
void print_edges(const vector<float>& data) {
    size_t n = data.size();
    cout << "First 10 elements: ";
    // Выводим минимум из размера массива и 10
    for (size_t i = 0; i < min(n, size_t(10)); i++)
        cout << data[i] << " ";
    cout << endl;
}

// ---------------------------
// Ядро: условно "некоалесцированный" доступ
// (здесь используется grid-stride loop, что само по себе эффективно, 
// но логика доступа отличается от прямого отображения tid -> i)
// ---------------------------
__global__ void kernel_noncoalesced(const float* data, float* output, int N) {
    // Вычисление уникального индекса потока
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // Шаг сетки (общее число потоков в запуске)
    int stride = blockDim.x * gridDim.x;
    // Цикл по массиву с шагом stride
    for (int i = tid; i < N; i += stride) {
        output[i] = data[i] * 2.0f; // Простая операция умножения
    }
}

// ---------------------------
// Ядро: классический коалесцированный доступ
// Потоки в варпе обращаются к соседним ячейкам памяти одновременно
// ---------------------------
__global__ void kernel_coalesced(const float* data, float* output, int N) {
    // Прямое вычисление глобального индекса
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // Проверка на выход за границы массива
    if (tid < N) {
        output[tid] = data[tid] * 2.0f;
    }
}

// ---------------------------
// Ядро с использованием разделяемой (shared) памяти
// Используется для ускорения доступа внутри блока
// ---------------------------
__global__ void kernel_shared(const float* data, float* output, int N) {
    // Объявление динамической разделяемой памяти
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int local_tid = threadIdx.x; // Индекс потока внутри текущего блока

    // Загрузка данных из глобальной памяти в разделяемую
    if (tid < N)
        sdata[local_tid] = data[tid];
    
    // Синхронизация всех потоков блока: ждем окончания загрузки в sdata
    __syncthreads();

    // Чтение из разделяемой памяти и запись результата в глобальную
    if (tid < N)
        output[tid] = sdata[local_tid] * 2.0f;
}

// ---------------------------
// Главная функция
// ---------------------------
int main() {
    // Наборы размеров массивов и конфигураций блоков для тестов
    vector<int> sizes = {10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000};
    vector<int> threads_per_block_list = {64, 128, 256, 512};

    // Настройка генератора случайных чисел
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> dist(RAND_MIN_VAL, RAND_MAX_VAL);

    // Основной цикл по размерам данных
    for (int N : sizes) {
        cout << "------------------------------------------";
        cout << endl << "Array size: " << N << endl;

        // Создание и заполнение вектора на хосте (CPU)
        vector<float> h_data(N);
        for (int i = 0; i < N; i++)
            h_data[i] = static_cast<float>(dist(gen));

        print_edges(h_data);

        // Выделение памяти на устройстве (GPU)
        float *d_data, *d_output;
        cudaMalloc(&d_data, N * sizeof(float));
        cudaMalloc(&d_output, N * sizeof(float));

        // Копирование данных с CPU на GPU
        cudaMemcpy(d_data, h_data.data(), N * sizeof(float), cudaMemcpyHostToDevice);

        // Вложенный цикл по количеству потоков в блоке
        for (int threads_per_block : threads_per_block_list) {
            // Расчет количества блоков, необходимых для покрытия всего массива
            int blocks = (N + threads_per_block - 1) / threads_per_block;

            // Создание событий CUDA для точного замера времени
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            // ---------------------------
            // ТЕСТ: Коалесцированный доступ (Global Memory)
            // ---------------------------
            cudaEventRecord(start); // Запуск таймера
            kernel_coalesced<<<blocks, threads_per_block>>>(d_data, d_output, N);
            cudaDeviceSynchronize(); // Ожидание завершения работы GPU
            cudaEventRecord(stop);  // Остановка таймера
            cudaEventSynchronize(stop);

            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop); // Расчет времени в мс

            // Копирование результата обратно на CPU для проверки
            vector<float> h_output(N);
            cudaMemcpy(h_output.data(), d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

            // Вычисление суммы, среднего и дисперсии на CPU для валидации
            float sum = 0, mean = 0, var = 0;
            for (float x : h_output) sum += x;
            mean = sum / N;
            for (float x : h_output) var += (x - mean) * (x - mean);
            var /= N;

            // Вывод метрик для коалесцированного доступа
            cout << "\nKernel: Coalesced (Global Memory)" << endl;
            cout << "Threads/block: " << threads_per_block << endl;
            cout << "Sum: " << sum << endl;
            cout << "Mean: " << mean << endl;
            cout << "Variance: " << var << endl;
            cout << "Time (ms): " << milliseconds << endl;

            // ---------------------------
            // ТЕСТ: Некоалесцированный доступ
            // ---------------------------
            cudaEventRecord(start);
            kernel_noncoalesced<<<blocks, threads_per_block>>>(d_data, d_output, N);
            cudaDeviceSynchronize();
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&milliseconds, start, stop);

            cudaMemcpy(h_output.data(), d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

            // Сброс и пересчет статистики
            sum = mean = var = 0;
            for (float x : h_output) sum += x;
            mean = sum / N;
            for (float x : h_output) var += (x - mean) * (x - mean);
            var /= N;

            cout << "\nKernel: Non-Coalesced (Global Memory)" << endl;
            cout << "Threads/block: " << threads_per_block << endl;
            cout << "Sum: " << sum << endl;
            cout << "Mean: " << mean << endl;
            cout << "Variance: " << var << endl;
            cout << "Time (ms): " << milliseconds << endl;

            // ---------------------------
            // ТЕСТ: Использование Shared memory
            // ---------------------------
            // Определение размера динамической shared-памяти в байтах
            size_t shared_mem_size = threads_per_block * sizeof(float);
            cudaEventRecord(start);
            // Запуск ядра с передачей размера shared-памяти третьим параметром
            kernel_shared<<<blocks, threads_per_block, shared_mem_size>>>(d_data, d_output, N);
            cudaDeviceSynchronize();
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&milliseconds, start, stop);

            cudaMemcpy(h_output.data(), d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

            sum = mean = var = 0;
            for (float x : h_output) sum += x;
            mean = sum / N;
            for (float x : h_output) var += (x - mean) * (x - mean);
            var /= N;

            cout << "\nKernel: Coalesced (Shared Memory)" << endl;
            cout << "Threads/block: " << threads_per_block << endl;
            cout << "Sum: " << sum << endl;
            cout << "Mean: " << mean << endl;
            cout << "Variance: " << var << endl;
            cout << "Time (ms): " << milliseconds << endl;
            
            // Удаление событий для текущей итерации
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }

        // Освобождение памяти на GPU перед переходом к следующему размеру N
        cudaFree(d_data);
        cudaFree(d_output);
    }

    return 0;
}
