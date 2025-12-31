#include <iostream>      // Стандартный ввод / вывод
#include <vector>        // Контейнер vector
#include <algorithm>     // min, max
#include <cuda_runtime.h>// CUDA Runtime API
#include <random>        // Генерация случайных чисел
#include <chrono>        // Измерение времени выполнения
#include <omp.h>         // OpenMP для параллелизма на CPU

using namespace std;

// ======================================================
// CUDA Kernel
// Назначение:
//   Каждый CUDA-блок сортирует свой подмассив фиксированного
//   размера, используя shared memory.
// Подход:
//   1) Копирование данных из global memory в shared memory
//   2) Сортировка внутри блока (Insertion Sort)
//   3) Запись отсортированных данных обратно в global memory
// ======================================================
__global__ void sortBlocksSharedKernel(int* data, int blockSize, int n) {

    // Выделяем shared memory динамически
    // Размер задаётся при запуске kernel
    extern __shared__ int s[];

    // Локальный индекс потока в блоке
    int tid = threadIdx.x;

    // Глобальный индекс элемента в массиве
    int gid = blockIdx.x * blockSize + tid;

    // Защита от выхода за границы массива
    if (gid < n)
        s[tid] = data[gid];
    else
        // Если элементов меньше, чем размер блока,
        // заполняем "лишние" элементы максимально возможным значением
        s[tid] = INT_MAX;

    // Синхронизация: все потоки должны загрузить данные
    __syncthreads();

    // --------------------------------------------------
    // Сортировка вставками (Insertion Sort) внутри блока
    // Подходит для небольших массивов (shared memory)
    // --------------------------------------------------
    for (int i = 1; i < blockSize; i++) {
        int key = s[i];
        int j = i - 1;

        // Сдвигаем элементы вправо, пока не найдём позицию
        while (j >= 0 && s[j] > key) {
            s[j + 1] = s[j];
            j--;
        }
        s[j + 1] = key;

        // Синхронизация после каждого шага
        __syncthreads();
    }

    // Записываем отсортированные данные обратно в global memory
    if (gid < n)
        data[gid] = s[tid];
}

// ======================================================
// CPU-функция слияния двух отсортированных подмассивов
// [left, mid) и [mid, right)
// Используется в Merge Sort
// ======================================================
void merge(int* arr, int* temp, int left, int mid, int right) {
    int i = left;   // указатель левой части
    int j = mid;    // указатель правой части
    int k = left;   // указатель во временный массив

    // Основное слияние
    while (i < mid && j < right)
        temp[k++] = (arr[i] <= arr[j]) ? arr[i++] : arr[j++];

    // Копирование оставшихся элементов
    while (i < mid)   temp[k++] = arr[i++];
    while (j < right) temp[k++] = arr[j++];
}

// ======================================================
// Параллельная версия Merge Sort с использованием OpenMP
// Алгоритм:
//   Bottom-up Merge Sort
//   На каждом шаге увеличиваем ширину сливаемых блоков
// ======================================================
void parallelMergeSortOMP(int* arr, int n) {

    // Временный массив для слияния
    int* temp = new int[n];

    // Ширина подмассивов: 1, 2, 4, 8, ...
    for (int width = 1; width < n; width *= 2) {

        // Параллельное слияние подмассивов
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i += 2 * width) {
            int left  = i;
            int mid   = min(i + width, n);
            int right = min(i + 2 * width, n);

            merge(arr, temp, left, mid, right);
        }

        // Параллельное копирование обратно в основной массив
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++)
            arr[i] = temp[i];
    }

    delete[] temp;
}

// ======================================================
// Проверка корректности сортировки
// ======================================================
bool isSorted(const vector<int>& v) {
    for (size_t i = 1; i < v.size(); i++)
        if (v[i - 1] > v[i])
            return false;
    return true;
}

// ======================================================
// Main
// ======================================================
int main() {

    // Размеры массивов для тестирования
    vector<int> sizes = {100, 1000, 10000, 100000, 1000000};

    // Генератор случайных чисел
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> dist(0, 1000000);

    // Тестируем для каждого размера
    for (int n : sizes) {

        // Инициализация массива на CPU
        vector<int> h(n);
        for (int& x : h)
            x = dist(gen);

        cout << "========== Размер массива: " << n << " ==========\n";

        // Вывод первых и последних элементов до сортировки
        cout << "До сортировки (первые 10): ";
        for (int i = 0; i < min(10, n); i++) cout << h[i] << " ";
        cout << "\nДо сортировки (последние 10): ";
        for (int i = max(0, n - 10); i < n; i++) cout << h[i] << " ";
        cout << "\n";

        // Выделение памяти на GPU
        int* d;
        cudaMalloc(&d, n * sizeof(int));

        // Копирование данных CPU → GPU
        cudaMemcpy(d, h.data(), n * sizeof(int), cudaMemcpyHostToDevice);

        // Конфигурация CUDA
        int blockSize = 1024; // максимальный размер блока
        int blocks = (n + blockSize - 1) / blockSize;

        // Замер времени
        auto start = chrono::high_resolution_clock::now();

        // --------------------------------------------------
        // Этап 1: CUDA
        // Сортировка локальных подмассивов на GPU
        // --------------------------------------------------
        sortBlocksSharedKernel<<<blocks, blockSize,
            blockSize * sizeof(int)>>>(d, blockSize, n);

        cudaDeviceSynchronize();

        // Копирование результата обратно на CPU
        cudaMemcpy(h.data(), d, n * sizeof(int), cudaMemcpyDeviceToHost);

        // --------------------------------------------------
        // Этап 2: CPU + OpenMP
        // Параллельное слияние отсортированных блоков
        // --------------------------------------------------
        parallelMergeSortOMP(h.data(), n);

        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double, milli> elapsed = end - start;

        // Вывод результата
        cout << "После сортировки (первые 10): ";
        for (int i = 0; i < min(10, n); i++) cout << h[i] << " ";
        cout << "\nПосле сортировки (последние 10): ";
        for (int i = max(0, n - 10); i < n; i++) cout << h[i] << " ";
        cout << "\n";

        cout << "Время сортировки: " << elapsed.count() << " мс\n";
        cout << (isSorted(h)
                 ? "Массив отсортирован корректно\n\n"
                 : "ОШИБКА СОРТИРОВКИ\n\n");

        // Освобождение памяти GPU
        cudaFree(d);
    }

    return 0;
}
