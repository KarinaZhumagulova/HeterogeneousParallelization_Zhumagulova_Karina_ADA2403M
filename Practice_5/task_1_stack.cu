%%writefile practice_work_5_stack.cu

#include <iostream>        // Стандартный ввод-вывод
#include <vector>          // Контейнер vector для работы на CPU
#include <cuda_runtime.h>  // Основные функции CUDA
#include <cstdlib>         // Стандартные функции (exit и др.)
#include <ctime>           // Работа со временем (для random)
#include <random>          // Генератор случайных чисел

using namespace std;

// Макрос для проверки ошибок CUDA: если функция вернула ошибку, выводим её и завершаем программу
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
const int MAX_STACK_SIZE = 10000000; // Максимальная емкость стека (10 млн элементов)
const int BLOCK_SIZE = 256;          // Количество потоков в одном блоке CUDA
const int INITIAL_STACK_SIZE = 1000000; // Начальное кол-во элементов в стеке

// ==============================
// Глобальный стек на GPU (в глобальной памяти видеокарты)
// ==============================
__device__ int d_stack[MAX_STACK_SIZE]; // Массив, представляющий стек
__device__ int d_top = 0;               // Индекс вершины стека (разделяемая переменная)

// ==============================
// Ядро инициализации стека
// ==============================
__global__ void init_stack_kernel(int n) {
    // Вычисляем глобальный индекс потока
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Заполняем начальные элементы стека значениями индекса
    if (idx < n) {
        d_stack[idx] = idx; 
    }
    // Один поток (нулевой) устанавливает начальное значение указателя вершины
    if (idx == 0) {
        d_top = n;
    }
}

// ==============================
// Ядро Push (Добавление в стек)
// ==============================
__global__ void push_kernel(int* values, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return; // Проверка выхода за границы входного массива

    // Атомарно увеличиваем d_top на 1 и получаем СТАРОЕ значение (индекс для вставки)
    int pos = atomicAdd(&d_top, 1);
    
    // Если место в стеке еще есть
    if (pos < MAX_STACK_SIZE) {
        d_stack[pos] = values[idx]; // Записываем значение в стек
    } else {
        // Если стек переполнен, возвращаем указатель обратно
        atomicSub(&d_top, 1);
    }
}

// ==============================
// Ядро Pop (Удаление из стека)
// ==============================
__global__ void pop_kernel(int* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Атомарно уменьшаем d_top на 1 и получаем СТАРОЕ значение
    int pos = atomicSub(&d_top, 1);
    
    // Если в стеке были элементы (pos > 0)
    if (pos > 0) {
        // Берем элемент по индексу (pos - 1)
        output[idx] = d_stack[pos - 1];
    } else {
        // Если стек был пуст, возвращаем указатель в 0
        atomicAdd(&d_top, 1);
        output[idx] = -1; // Сигнализируем, что извлечь ничего не удалось
    }
}

// ==============================
// Вспомогательная функция печати стека (выполняется на CPU)
// ==============================
void print_stack(const vector<int>& data, int top) {
    cout << "Stack size = " << top << "\n";

    // Вывод первых 10 элементов
    cout << "First 10: ";
    for (int i = 0; i < min(10, top); i++)
        cout << data[i] << " ";
    cout << "\n";

    // Вывод последних 10 элементов
    cout << "Last 10: ";
    for (int i = max(0, top - 10); i < top; i++)
        cout << data[i] << " ";
    cout << "\n\n";
}

// ==============================
// Главная функция
// ==============================
int main() {
    cout << "=== Parallel Stack (LIFO) on GPU ===\n";

    const int N = 10000; // Количество элементов для новых операций push/pop

    // Подготовка данных на CPU
    vector<int> h_values(N);
    mt19937 gen(time(nullptr)); // Генератор случайных чисел
    uniform_int_distribution<int> dist(1, 1000000);
    for (int i = 0; i < N; ++i)
        h_values[i] = dist(gen); // Заполнение случайными числами

    // Выделение памяти на GPU
    int *d_values, *d_output;
    CUDA_CHECK(cudaMalloc(&d_values, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_output, N * sizeof(int)));
    
    // Копирование входных данных на GPU
    CUDA_CHECK(cudaMemcpy(d_values, h_values.data(), N * sizeof(int), cudaMemcpyHostToDevice));

    // Настройка сетки потоков
    dim3 block(BLOCK_SIZE);
    dim3 grid_init((INITIAL_STACK_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // --- ИНИЦИАЛИЗАЦИЯ ---
    init_stack_kernel<<<grid_init, block>>>(INITIAL_STACK_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize()); // Ждем завершения

    vector<int> h_stack(MAX_STACK_SIZE);
    int h_top;

    // Копирование данных из __device__ символов (глобальных переменных GPU) в CPU
    CUDA_CHECK(cudaMemcpyFromSymbol(&h_top, d_top, sizeof(int)));
    CUDA_CHECK(cudaMemcpyFromSymbol(h_stack.data(), d_stack, MAX_STACK_SIZE * sizeof(int)));

    cout << "--- Before push ---\n";
    print_stack(h_stack, h_top);

    // --- PUSH ---
    cudaEvent_t start, stop; // События для замера времени
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start); // Запуск таймера
    push_kernel<<<grid, block>>>(d_values, N);
    cudaEventRecord(stop);  // Остановка таймера
    CUDA_CHECK(cudaDeviceSynchronize());

    float push_time;
    cudaEventElapsedTime(&push_time, start, stop); // Вычисление времени в мс

    // Получаем состояние после push
    CUDA_CHECK(cudaMemcpyFromSymbol(&h_top, d_top, sizeof(int)));
    CUDA_CHECK(cudaMemcpyFromSymbol(h_stack.data(), d_stack, MAX_STACK_SIZE * sizeof(int)));

    cout << "--- After push ---\n";
    print_stack(h_stack, h_top);
    cout << "Expected stack size: " << INITIAL_STACK_SIZE + N << "\n";
    cout << "Actual stack size:   " << h_top << "\n\n";

    // --- POP ---
    cudaEventRecord(start);
    pop_kernel<<<grid, block>>>(d_output, N);
    cudaEventRecord(stop);
    CUDA_CHECK(cudaDeviceSynchronize());

    float pop_time;
    cudaEventElapsedTime(&pop_time, start, stop);

    // Получаем итоговое состояние
    CUDA_CHECK(cudaMemcpyFromSymbol(&h_top, d_top, sizeof(int)));
    CUDA_CHECK(cudaMemcpyFromSymbol(h_stack.data(), d_stack, MAX_STACK_SIZE * sizeof(int)));

    cout << "--- After pop ---\n";
    print_stack(h_stack, h_top);

    // Итоговые результаты
    cout << "Push time: " << push_time << " ms\n";
    cout << "Pop time:  " << pop_time << " ms\n";

    // Освобождение ресурсов
    cudaFree(d_values);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
