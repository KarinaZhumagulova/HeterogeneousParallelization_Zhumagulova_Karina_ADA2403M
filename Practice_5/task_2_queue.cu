%%writefile practice_work_5_queue.cu

#include <iostream>        // Для вывода в консоль
#include <vector>          // Для работы с векторами на стороне CPU
#include <cuda_runtime.h>  // Основной API CUDA
#include <cstdlib>         // Для использования exit()
#include <ctime>           // Для инициализации генератора случайных чисел временем
#include <random>          // Для генерации случайных чисел

using namespace std;

// Макрос для автоматической проверки ошибок CUDA после вызовов функций API
#define CUDA_CHECK(err) do { \
    cudaError_t e = (err); \
    if (e != cudaSuccess) { \
        cerr << "CUDA Error: " << cudaGetErrorString(e) \
             << " at line " << __LINE__ << endl; \
        exit(1); \
    } \
} while(0)

// ==============================
// Параметры
// ==============================
const int MAX_QUEUE_SIZE = 10000000; // Максимально допустимый размер очереди (10 млн)
const int BLOCK_SIZE = 256;          // Стандартный размер блока потоков
const int INITIAL_QUEUE_SIZE = 1000000; // Количество элементов при начальном заполнении

// ==============================
// Глобальная очередь на GPU
// ==============================
__device__ int d_queue[MAX_QUEUE_SIZE]; // Статический массив в глобальной памяти GPU
__device__ int d_head = 0;               // Указатель на начало очереди (откуда извлекаем)
__device__ int d_tail = 0;               // Указатель на конец очереди (куда добавляем)

// ==============================
// Инициализация очереди
// ==============================
__global__ void init_queue_kernel(int n) {
    // Вычисляем уникальный индекс потока
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Заполняем начальный участок очереди значениями индекса
    if (idx < n) {
        d_queue[idx] = idx; 
    }
    // Только первый поток сбрасывает указатели в начальное состояние
    if (idx == 0) {
        d_head = 0;
        d_tail = n;
    }
}

// ==============================
// Enqueue (добавление в хвост)
// ==============================
__global__ void enqueue_kernel(int* values, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return; // Если индекс потока больше количества входных данных — выходим

    // Атомарно увеличиваем хвост и получаем индекс для записи текущим потоком
    int pos = atomicAdd(&d_tail, 1);
    
    // Если в массиве еще есть место
    if (pos < MAX_QUEUE_SIZE) {
        d_queue[pos] = values[idx]; // Записываем значение из входного массива в очередь
    } else {
        // Если места нет — откатываем инкремент хвоста назад
        atomicSub(&d_tail, 1); 
    }
}

// ==============================
// Dequeue (извлечение из головы)
// ==============================
__global__ void dequeue_kernel(int* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Атомарно увеличиваем голову и получаем индекс элемента для извлечения
    int pos = atomicAdd(&d_head, 1);
    
    // Проверяем, не обогнала ли голова хвост (не пуста ли очередь)
    if (pos < d_tail) {
        output[idx] = d_queue[pos]; // Читаем значение из очереди в выходной массив
    } else {
        // Если очередь пуста — откатываем инкремент головы назад
        atomicSub(&d_head, 1); 
        output[idx] = -1; // Возвращаем -1 как признак пустоты
    }
}

// ==============================
// Вспомогательная печать очереди (CPU)
// ==============================
void print_queue(const vector<int>& data, int head, int tail) {
    int size = tail - head; // Текущее количество элементов
    cout << "Queue size = " << size << "\n";

    // Печатаем первые 10 актуальных элементов (начиная с head)
    cout << "First 10: ";
    for (int i = head; i < min(head + 10, tail); i++)
        cout << data[i] << " ";
    cout << "\n";

    // Печатаем последние 10 актуальных элементов (перед tail)
    cout << "Last 10: ";
    for (int i = max(head, tail - 10); i < tail; i++)
        cout << data[i] << " ";
    cout << "\n\n";
}

// ==============================
// MAIN
// ==============================
int main() {
    cout << "=== Parallel Queue (FIFO) on GPU ===\n";

    const int N = 10000; // Количество элементов для операций Enqueue/Dequeue

    // Подготовка случайных данных на хосте (CPU)
    vector<int> h_values(N);
    mt19937 gen(time(nullptr));
    uniform_int_distribution<int> dist(1, 1000000);
    for (int i = 0; i < N; ++i)
        h_values[i] = dist(gen);

    // Выделение памяти на видеокарте
    int *d_values, *d_output;
    CUDA_CHECK(cudaMalloc(&d_values, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_output, N * sizeof(int)));
    
    // Копирование данных для добавления на GPU
    CUDA_CHECK(cudaMemcpy(d_values, h_values.data(), N * sizeof(int), cudaMemcpyHostToDevice));

    // Параметры запуска ядер
    dim3 block(BLOCK_SIZE);
    dim3 grid_init((INITIAL_QUEUE_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // --- 1. ИНИЦИАЛИЗАЦИЯ ---
    init_queue_kernel<<<grid_init, block>>>(INITIAL_QUEUE_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize()); // Ждем завершения ядра

    vector<int> h_queue(MAX_QUEUE_SIZE);
    int h_head, h_tail;

    // Копируем значения переменных из памяти GPU в память CPU для проверки
    CUDA_CHECK(cudaMemcpyFromSymbol(&h_head, d_head, sizeof(int)));
    CUDA_CHECK(cudaMemcpyFromSymbol(&h_tail, d_tail, sizeof(int)));
    CUDA_CHECK(cudaMemcpyFromSymbol(h_queue.data(), d_queue, MAX_QUEUE_SIZE * sizeof(int)));

    cout << "--- Before enqueue ---\n";
    print_queue(h_queue, h_head, h_tail);

    // --- 2. ENQUEUE (ДОБАВЛЕНИЕ) ---
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start); // Замеряем время
    enqueue_kernel<<<grid, block>>>(d_values, N);
    cudaEventRecord(stop);
    CUDA_CHECK(cudaDeviceSynchronize());

    float enqueue_time;
    cudaEventElapsedTime(&enqueue_time, start, stop);

    // Обновляем локальные копии данных очереди
    CUDA_CHECK(cudaMemcpyFromSymbol(&h_head, d_head, sizeof(int)));
    CUDA_CHECK(cudaMemcpyFromSymbol(&h_tail, d_tail, sizeof(int)));
    CUDA_CHECK(cudaMemcpyFromSymbol(h_queue.data(), d_queue, MAX_QUEUE_SIZE * sizeof(int)));

    cout << "--- After enqueue ---\n";
    print_queue(h_queue, h_head, h_tail);
    cout << "Expected queue size: " << INITIAL_QUEUE_SIZE + N << "\n";
    cout << "Actual queue size:   " << (h_tail - h_head) << "\n\n";

    // --- 3. DEQUEUE (ИЗВЛЕЧЕНИЕ) ---
    cudaEventRecord(start);
    dequeue_kernel<<<grid, block>>>(d_output, N);
    cudaEventRecord(stop);
    CUDA_CHECK(cudaDeviceSynchronize());

    float dequeue_time;
    cudaEventElapsedTime(&dequeue_time, start, stop);

    // Обновляем данные для финальной печати
    CUDA_CHECK(cudaMemcpyFromSymbol(&h_head, d_head, sizeof(int)));
    CUDA_CHECK(cudaMemcpyFromSymbol(&h_tail, d_tail, sizeof(int)));
    CUDA_CHECK(cudaMemcpyFromSymbol(h_queue.data(), d_queue, MAX_QUEUE_SIZE * sizeof(int)));

    cout << "--- After dequeue ---\n";
    print_queue(h_queue, h_head, h_tail);

    // Вывод статистики производительности
    cout << "Enqueue time: " << enqueue_time << " ms\n";
    cout << "Dequeue time: " << dequeue_time << " ms\n";

    // Освобождение памяти и уничтожение событий
    cudaFree(d_values);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
