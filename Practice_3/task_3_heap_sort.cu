%%writefile task_3_heap_sort.cu
// ======================================================
// Parallel GPU Heap Sort (hybrid block heaps)
// ======================================================

// Подключаем стандартные библиотеки
#include <iostream>      // Для ввода/вывода (cout, cerr)
#include <vector>        // Для динамических массивов std::vector
#include <random>        // Для генерации случайных чисел
#include <chrono>        // Для измерения времени выполнения
#include <cuda_runtime.h> // Основной заголовок CUDA для работы с GPU
#include <algorithm>     // Для std::swap и std::min

using namespace std;    // Чтобы не писать std:: перед каждым объектом/функцией

// ======================================================
// Макрос для проверки ошибок CUDA
// ======================================================
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; /* Выполняем вызов CUDA */ \
    if (err != cudaSuccess) { /* Если есть ошибка */ \
        cerr << "CUDA error: " << cudaGetErrorString(err) /* Выводим сообщение об ошибке */ \
             << " at line " << __LINE__ << endl; /* Указываем строку */ \
        exit(EXIT_FAILURE); /* Завершаем программу */ \
    } \
} while(0)

// ======================================================
// GPU kernel: max-heapify для сегмента блока
// ======================================================
__global__ void heapify_block(int *arr, int n, int blockSize) {
    int blockStart = blockIdx.x * blockSize;               // Начальный индекс блока
    int blockEnd = min(blockStart + blockSize, n);        // Конечный индекс блока (не выходим за предел массива)

    // Цикл для heapify каждого узла внутри блока
    // В данном случае один поток проходит весь блок
    for (int i = blockEnd/2 - 1; i >= blockStart; --i) {
        int largest = i;                                  // Изначально считаем текущий узел максимальным
        int l = 2*(i - blockStart) + 1 + blockStart;     // Индекс левого потомка в блоке
        int r = 2*(i - blockStart) + 2 + blockStart;     // Индекс правого потомка в блоке

        if (l < blockEnd && arr[l] > arr[largest]) largest = l; // Если левый потомок больше — обновляем
        if (r < blockEnd && arr[r] > arr[largest]) largest = r; // Если правый потомок больше — обновляем

        if (largest != i) {                               // Если корень не является наибольшим
            int tmp = arr[i];                             // Меняем элементы местами
            arr[i] = arr[largest];
            arr[largest] = tmp;
        }
    }
}

// ======================================================
// CPU Heapify для полной сортировки
// ======================================================
void heapify_cpu(vector<int>& arr, int n, int i) {
    int largest = i;          // Изначально считаем текущий элемент максимальным
    int l = 2*i + 1;          // Индекс левого потомка
    int r = 2*i + 2;          // Индекс правого потомка

    if (l < n && arr[l] > arr[largest]) largest = l; // Проверка левого потомка
    if (r < n && arr[r] > arr[largest]) largest = r; // Проверка правого потомка

    if (largest != i) {       // Если наибольший не корень
        swap(arr[i], arr[largest]);       // Меняем местами
        heapify_cpu(arr, n, largest);     // Рекурсивно продолжаем heapify
    }
}

// ======================================================
// CPU HeapSort для слияния локальных куч
// ======================================================
void heapSort_cpu(vector<int>& arr) {
    int n = arr.size();                  // Размер массива

    // Построение максимальной кучи
    for (int i = n/2 -1; i>=0; --i)
        heapify_cpu(arr, n, i);          // Heapify каждого родителя

    // Последовательное извлечение максимума
    for (int i=n-1; i>0; --i) {
        swap(arr[0], arr[i]);            // Меняем корень с последним элементом
        heapify_cpu(arr, i, 0);          // Heapify для оставшейся части массива
    }
}

// ======================================================
// Гибридная GPU Heap Sort (локальные кучки на GPU + окончательная сортировка на CPU)
// ======================================================
void hybridHeapSort(vector<int>& arr) {
    int n = arr.size();       // Размер массива
    int *d_arr = nullptr;     // Указатель на массив в памяти GPU

    CUDA_CHECK(cudaMalloc(&d_arr, n * sizeof(int)));                  // Выделяем память на GPU
    CUDA_CHECK(cudaMemcpy(d_arr, arr.data(), n * sizeof(int), cudaMemcpyHostToDevice)); // Копируем данные на GPU

    int blockSize = 1024;                    // Размер блока (количество элементов на блок)
    dim3 threads(1);                         // Один поток на блок
    dim3 blocks((n + blockSize -1)/blockSize); // Вычисляем количество блоков

    // Параллельное построение локальных куч в блоках
    heapify_block<<<blocks, threads>>>(d_arr, n, blockSize);
    CUDA_CHECK(cudaDeviceSynchronize());     // Ждем завершения всех потоков

    // Копируем результат обратно на CPU
    CUDA_CHECK(cudaMemcpy(arr.data(), d_arr, n * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_arr));             // Освобождаем память GPU

    // Полная сортировка на CPU (слияние локальных куч)
    heapSort_cpu(arr);
}

// ======================================================
// Функция вывода первых и последних 10 элементов массива
// ======================================================
void printEdges(const vector<int>& arr) {
    int n = arr.size();                     // Размер массива
    cout << "Первые 10: ";
    for (int i=0;i<min(10,n);++i)          // Вывод первых 10 элементов
        cout<<arr[i]<<" ";
    cout<<"\nПоследние 10: ";
    for (int i=max(0,n-10); i<n;++i)       // Вывод последних 10 элементов
        cout<<arr[i]<<" ";
    cout<<"\n";
}

// ======================================================
// Проверка, отсортирован ли массив
// ======================================================
bool isSorted(const vector<int>& arr) {
    for (size_t i=1;i<arr.size();++i)
        if (arr[i-1]>arr[i]) return false; // Если хотя бы один элемент нарушает порядок — массив не отсортирован
    return true;                           // Иначе — отсортирован
}

// ======================================================
// Главная функция
// ======================================================
int main() {
    vector<int> testSizes = {100, 1000, 10000, 100000, 1000000}; // Размеры тестовых массивов

    mt19937 gen(random_device{}());           // Генератор случайных чисел
    uniform_int_distribution<int> dist(-100000,100000); // Диапазон значений

    for (int n:testSizes) {                   // Проходим по каждому размеру массива
        vector<int> arr(n);                   // Создаем массив
        for (int &x:arr) x=dist(gen);         // Заполняем случайными числами

        cout<<"====================================\n";
        cout<<"Размер массива: "<<n<<"\n";
        cout<<"До сортировки:\n";
        printEdges(arr);                      // Вывод первых и последних 10 элементов до сортировки

        auto start = chrono::high_resolution_clock::now(); // Начало измерения времени

        // ==== Гибридная GPU Heap Sort ====
        hybridHeapSort(arr);

        auto end = chrono::high_resolution_clock::now();   // Конец измерения времени
        chrono::duration<double,milli> elapsed = end-start; // Вычисляем длительность в миллисекундах

        cout<<"После сортировки:\n";
        printEdges(arr);                      // Вывод первых и последних 10 элементов после сортировки
        cout<<"Время выполнения: "<<elapsed.count()<<" мс\n"; // Вывод времени
        cout<<"Массив отсортирован: "<<(isSorted(arr)?"ДА":"НЕТ")<<"\n"; // Проверка правильности сортировки
    }

    return 0; // Успешное завершение программы
}
