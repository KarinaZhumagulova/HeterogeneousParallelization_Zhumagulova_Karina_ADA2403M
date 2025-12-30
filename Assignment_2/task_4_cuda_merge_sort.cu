#include <iostream>        // Стандартная библиотека ввода-вывода
#include <vector>          // Контейнер vector
#include <algorithm>       // Стандартные алгоритмы (min, max)
#include <cuda_runtime.h>  // CUDA Runtime API
#include <random>          // Генерация случайных чисел
#include <chrono>          // Измерение времени выполнения
#include <omp.h>           // OpenMP для параллельных вычислений на CPU

using namespace std;       // Использование стандартного пространства имён

// ======================================================
// CUDA kernel: сортировка подмассива внутри одного CUDA-блока
// Используется shared memory для ускорения доступа
// ======================================================
__global__ void sortBlocksSharedKernel(int* data, int blockSize, int n) {

    extern __shared__ int s[];                 // Динамическая shared memory

    int tid = threadIdx.x;                     // Локальный индекс потока в блоке
    int gid = blockIdx.x * blockSize + tid;    // Глобальный индекс элемента

    if (gid < n)
        s[tid] = data[gid];                    // Копирование данных в shared memory
    else
        s[tid] = INT_MAX;                      // Заполнение фиктивным значением

    __syncthreads();                           // Синхронизация потоков блока

    for (int i = 1; i < blockSize; i++) {      // Сортировка вставками
        int key = s[i];                        // Текущий элемент
        int j = i - 1;                         // Индекс предыдущего элемента

        while (j >= 0 && s[j] > key) {          // Сдвиг элементов вправо
            s[j + 1] = s[j];
            j--;
        }

        s[j + 1] = key;                        // Вставка элемента
        __syncthreads();                       // Синхронизация потоков
    }

    if (gid < n)
        data[gid] = s[tid];                    // Запись результата в глобальную память
}

// ======================================================
// Слияние двух отсортированных подмассивов (CPU)
// ======================================================
void merge(int* arr, int* temp, int left, int mid, int right) {

    int i = left;      // Индекс левого подмассива
    int j = mid;       // Индекс правого подмассива
    int k = left;      // Индекс временного массива

    while (i < mid && j < right)
        temp[k++] = (arr[i] <= arr[j]) ? arr[i++] : arr[j++]; // Слияние

    while (i < mid) temp[k++] = arr[i++];   // Копирование остатка слева
    while (j < right) temp[k++] = arr[j++]; // Копирование остатка справа
}

// ======================================================
// Параллельная сортировка слиянием с использованием OpenMP
// ======================================================
void parallelMergeSortOMP(int* arr, int n) {

    int* temp = new int[n];                   // Временный массив

    for (int width = 1; width < n; width *= 2) {

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i += 2 * width) {
            int left  = i;                    // Левая граница
            int mid   = min(i + width, n);    // Середина
            int right = min(i + 2 * width, n);// Правая граница
            merge(arr, temp, left, mid, right);// Слияние
        }

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++)
            arr[i] = temp[i];                 // Копирование результата
    }

    delete[] temp;                            // Освобождение памяти
}

// ======================================================
// Проверка корректности сортировки
// ======================================================
bool isSorted(const vector<int>& v) {

    for (size_t i = 1; i < v.size(); i++)
        if (v[i - 1] > v[i]) return false;    // Нарушение порядка

    return true;                              // Массив отсортирован
}

// ======================================================
// Главная функция
// ======================================================
int main() {

    vector<int> sizes = {100, 1000, 10000, 100000, 1000000}; // Размеры массивов

    random_device rd;                         // Источник энтропии
    mt19937 gen(rd());                        // Генератор Mersenne Twister
    uniform_int_distribution<int> dist(0, 1000000); // Диапазон чисел

    for (int n : sizes) {

        vector<int> h(n);                     // Массив на CPU
        for (int& x : h) x = dist(gen);       // Заполнение случайными числами

        cout << "========== Размер массива: " << n << " ==========\n";
        cout << "До сортировки (первые 10): ";
        for (int i = 0; i < min(10, n); i++) cout << h[i] << " ";

        cout << "\nДо сортировки (последние 10): ";
        for (int i = max(0, n - 10); i < n; i++) cout << h[i] << " ";
        cout << "\n";

        int* d;                               // Указатель на память GPU
        cudaMalloc(&d, n * sizeof(int));      // Выделение памяти на GPU
        cudaMemcpy(d, h.data(), n * sizeof(int),
                   cudaMemcpyHostToDevice);   // Копирование на GPU

        int blockSize = 1024;                 // Размер CUDA-блока
        int blocks = (n + blockSize - 1) / blockSize; // Количество блоков

        auto start = chrono::high_resolution_clock::now(); // Старт таймера

        sortBlocksSharedKernel<<<blocks, blockSize,
            blockSize * sizeof(int)>>>(d, blockSize, n);   // CUDA сортировка

        cudaDeviceSynchronize();              // Ожидание GPU

        cudaMemcpy(h.data(), d, n * sizeof(int),
                   cudaMemcpyDeviceToHost);   // Копирование обратно на CPU

        parallelMergeSortOMP(h.data(), n);    // Параллельное слияние (CPU)

        auto end = chrono::high_resolution_clock::now();   // Конец таймера
        chrono::duration<double, milli> elapsed = end - start;

        cout << "После сортировки (первые 10): ";
        for (int i = 0; i < min(10, n); i++) cout << h[i] << " ";

        cout << "\nПосле сортировки (последние 10): ";
        for (int i = max(0, n - 10); i < n; i++) cout << h[i] << " ";
        cout << "\n";

        cout << "Время сортировки: " << elapsed.count() << " мс\n";
        cout << (isSorted(h)
            ? "Массив отсортирован корректно\n\n"
            : "ОШИБКА СОРТИРОВКИ\n\n");

        cudaFree(d);                          // Освобождение памяти GPU
    }

    return 0;                                 // Завершение программы
}
