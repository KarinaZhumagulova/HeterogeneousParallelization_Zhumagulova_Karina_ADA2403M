%%writefile task_2_quick_sort_v2.cu
// ======================================================
// Гибридная быстрая сортировка (Quick Sort):
// подсчёт элементов на GPU + разбиение и рекурсия на CPU
// ======================================================

// Подключение стандартной библиотеки ввода-вывода
#include <iostream>

// Подключение контейнера vector
#include <vector>

// Подключение стандартных алгоритмов (swap, min, max)
#include <algorithm>

// Подключение генератора случайных чисел
#include <random>

// Подключение библиотеки для измерения времени
#include <chrono>

// Подключение CUDA Runtime API
#include <cuda_runtime.h>

// Используем стандартное пространство имён
using namespace std;

// ======================================================
// Макрос для проверки ошибок CUDA
// ======================================================
// Проверяет результат CUDA-вызова и завершает программу
// при возникновении ошибки
#define CUDA_SAFE(call)                                      \
    do {                                                     \
        cudaError_t err = call;                              \
        if (err != cudaSuccess) {                            \
            cerr << "CUDA error: "                           \
                 << cudaGetErrorString(err)                  \
                 << " at line " << __LINE__ << endl;         \
            exit(EXIT_FAILURE);                              \
        }                                                    \
    } while (0)

// ======================================================
// CUDA-ядро: подсчёт элементов меньше опорного
// ======================================================
// d_array   — массив данных на устройстве (GPU)
// pivot     — опорный элемент
// d_counter — счётчик элементов меньше pivot
// size      — размер обрабатываемого массива
__global__ void gpuCountLower(const int* d_array,
                              int pivot,
                              int* d_counter,
                              int size) {

    // Глобальный индекс потока
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Проверяем, что индекс не выходит за пределы массива
    // и элемент меньше опорного
    if (tid < size && d_array[tid] < pivot) {
        // Атомарно увеличиваем счётчик
        atomicAdd(d_counter, 1);
    }
}

// ======================================================
// Гибридная быстрая сортировка (Quick Sort)
// ======================================================
// data  — сортируемый вектор
// left  — левая граница
// right — правая граница
void hybridQuickSort(vector<int>& data, int left, int right) {

    // Базовый случай рекурсии
    if (left >= right) return;

    // Выбор опорного элемента (последний элемент)
    int pivotValue = data[right];

    // Размер текущего сегмента
    int segmentSize = right - left + 1;

    // Указатели на память GPU
    int* d_segment = nullptr;
    int* d_lessCount = nullptr;

    // Выделение памяти на GPU для сегмента массива
    CUDA_SAFE(cudaMalloc(&d_segment, segmentSize * sizeof(int)));

    // Выделение памяти на GPU для счётчика
    CUDA_SAFE(cudaMalloc(&d_lessCount, sizeof(int)));

    // Копирование сегмента массива с CPU на GPU
    CUDA_SAFE(cudaMemcpy(d_segment,
                         &data[left],
                         segmentSize * sizeof(int),
                         cudaMemcpyHostToDevice));

    // Обнуление счётчика на GPU
    CUDA_SAFE(cudaMemset(d_lessCount, 0, sizeof(int)));

    // Настройка конфигурации CUDA
    dim3 threads(256);  // количество потоков в блоке
    dim3 blocks((segmentSize + threads.x - 1) / threads.x); // количество блоков

    // Запуск CUDA-ядра для подсчёта элементов
    gpuCountLower<<<blocks, threads>>>(d_segment,
                                       pivotValue,
                                       d_lessCount,
                                       segmentSize);

    // Ожидание завершения выполнения ядра
    CUDA_SAFE(cudaDeviceSynchronize());

    // ---- Разбиение массива на CPU ----
    int storeIndex = left - 1;

    // Проход по массиву и перестановка элементов
    for (int j = left; j < right; ++j) {
        if (data[j] < pivotValue) {
            ++storeIndex;
            swap(data[storeIndex], data[j]);
        }
    }

    // Перемещение опорного элемента на своё место
    swap(data[storeIndex + 1], data[right]);

    // Индекс опорного элемента после разбиения
    int pivotIndex = storeIndex + 1;

    // Освобождение памяти GPU
    CUDA_SAFE(cudaFree(d_segment));
    CUDA_SAFE(cudaFree(d_lessCount));

    // Рекурсивная сортировка левой части
    hybridQuickSort(data, left, pivotIndex - 1);

    // Рекурсивная сортировка правой части
    hybridQuickSort(data, pivotIndex + 1, right);
}

// ======================================================
// Проверка, отсортирован ли массив
// ======================================================
bool checkSorted(const vector<int>& data) {

    // Проверяем, что каждый элемент не больше следующего
    for (size_t i = 1; i < data.size(); ++i)
        if (data[i - 1] > data[i])
            return false;

    return true;
}

// ======================================================
// Вывод первых и последних 10 элементов массива
// ======================================================
void printEdges(const vector<int>& data) {

    // Размер массива
    int n = data.size();

    // Вывод первых 10 элементов
    cout << "Первые 10: ";
    for (int i = 0; i < min(10, n); ++i)
        cout << data[i] << " ";

    // Вывод последних 10 элементов
    cout << "\nПоследние 10: ";
    for (int i = max(0, n - 10); i < n; ++i)
        cout << data[i] << " ";

    cout << "\n";
}

// ======================================================
// Главная функция
// ======================================================
int main() {

    // Набор размеров массивов для тестирования
    vector<int> testSizes = {100, 1000, 10000, 100000, 1000000};

    // Инициализация генератора случайных чисел
    mt19937 rng(random_device{}());

    // Диапазон случайных значений
    uniform_int_distribution<int> dist(-100000, 100000);

    // Цикл по всем размерам массивов
    for (int size : testSizes) {

        // Создание массива заданного размера
        vector<int> values(size);

        // Заполнение массива случайными числами
        for (int& v : values)
            v = dist(rng);

        // Вывод разделителя
        cout << "====================================\n";

        // Вывод размера массива
        cout << "Размер массива: " << size << "\n";

        // Вывод массива до сортировки
        cout << "До сортировки:\n";
        printEdges(values);

        // Засечение времени начала
        auto startTime = chrono::high_resolution_clock::now();

        // Запуск гибридной сортировки
        hybridQuickSort(values, 0, size - 1);

        // Засечение времени окончания
        auto endTime = chrono::high_resolution_clock::now();

        // Вычисление времени выполнения
        chrono::duration<double, milli> elapsed = endTime - startTime;

        // Вывод массива после сортировки
        cout << "После сортировки:\n";
        printEdges(values);

        // Вывод времени выполнения
        cout << "Время выполнения: "
             << elapsed.count() << " мс\n";

        // Проверка корректности сортировки
        cout << "Массив отсортирован: "
             << (checkSorted(values) ? "ДА" : "НЕТ") << "\n";
    }

    // Завершение программы
    return 0;
}
