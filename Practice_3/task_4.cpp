#include <iostream>      // Для ввода/вывода (cout, cerr)
#include <vector>        // Для работы с динамическими массивами std::vector
#include <random>        // Для генерации случайных чисел
#include <chrono>        // Для измерения времени выполнения
#include <omp.h>         // Для работы с OpenMP (параллельные вычисления)
#include <algorithm>     // Для std::swap и std::copy

using namespace std;    // Чтобы не писать std:: перед каждым объектом/функцией

// ======================================================
// Константы для генерации случайных чисел
// ======================================================
constexpr int RAND_MIN_VAL = -1000;  // Минимальное значение для массива
constexpr int RAND_MAX_VAL = 1000;   // Максимальное значение для массива

// ======================================================
// Функция проверки, отсортирован ли массив
// ======================================================
bool isSorted(int* arr, int size) {
    for (int i = 0; i < size - 1; i++) {       // Проходим по всем элементам
        if (arr[i] > arr[i + 1])               // Если текущий элемент больше следующего
            return false;                      // Массив не отсортирован
    }
    return true;                               // Иначе массив отсортирован
}

// ======================================================
// Последовательные алгоритмы сортировки
// ======================================================

// ---------- Merge Sort ----------

// Функция слияния двух отсортированных подмассивов
void merge(int* arr, int l, int m, int r) {
    int n1 = m - l + 1;                       // Размер левого подмассива
    int n2 = r - m;                           // Размер правого подмассива

    vector<int> L(n1), R(n2);                 // Временные массивы для слияния
    for (int i = 0; i < n1; i++) L[i] = arr[l + i];        // Копируем левый подмассив
    for (int j = 0; j < n2; j++) R[j] = arr[m + 1 + j];    // Копируем правый подмассив

    int i = 0, j = 0, k = l;                  // Индексы для L, R и исходного массива
    while (i < n1 && j < n2) {                // Пока оба подмассива не закончились
        if (L[i] <= R[j])                     // Берем меньший элемент
            arr[k++] = L[i++];
        else
            arr[k++] = R[j++];
    }
    while (i < n1) arr[k++] = L[i++];         // Копируем оставшиеся элементы из L
    while (j < n2) arr[k++] = R[j++];         // Копируем оставшиеся элементы из R
}

// Рекурсивная реализация последовательного Merge Sort
void mergeSortSequential(int* arr, int l, int r) {
    if (l < r) {                              // Пока массив содержит больше одного элемента
        int m = l + (r - l) / 2;              // Индекс середины
        mergeSortSequential(arr, l, m);       // Сортируем левую половину
        mergeSortSequential(arr, m + 1, r);   // Сортируем правую половину
        merge(arr, l, m, r);                  // Сливаем отсортированные половины
    }
}

// ---------- Quick Sort ----------

// Функция разбиения для Quick Sort
int partition(int* arr, int low, int high) {
    int pivot = arr[high];                     // Опорный элемент — последний элемент массива
    int i = low - 1;                           // Индекс меньшего элемента
    for (int j = low; j < high; j++) {        // Проходим по всем элементам
        if (arr[j] < pivot) {                 // Если элемент меньше опорного
            i++;                               // Увеличиваем индекс меньшего
            swap(arr[i], arr[j]);             // Меняем местами
        }
    }
    swap(arr[i + 1], arr[high]);               // Ставим pivot на правильное место
    return i + 1;                              // Возвращаем индекс pivot
}

// Рекурсивная последовательная реализация Quick Sort
void quickSortSequential(int* arr, int low, int high) {
    if (low < high) {                          // Пока есть элементы для сортировки
        int pi = partition(arr, low, high);    // Получаем индекс pivot
        quickSortSequential(arr, low, pi - 1); // Сортируем левую часть
        quickSortSequential(arr, pi + 1, high);// Сортируем правую часть
    }
}

// ---------- Heap Sort ----------

// Функция heapify для Heap Sort
void heapify(int* arr, int n, int i) {
    int largest = i;                            // Считаем текущий узел наибольшим
    int l = 2 * i + 1;                          // Левый потомок
    int r = 2 * i + 2;                          // Правый потомок

    if (l < n && arr[l] > arr[largest]) largest = l; // Проверка левого потомка
    if (r < n && arr[r] > arr[largest]) largest = r; // Проверка правого потомка

    if (largest != i) {                          // Если наибольший не корень
        swap(arr[i], arr[largest]);             // Меняем местами
        heapify(arr, n, largest);               // Рекурсивно heapify для поддерева
    }
}

// Последовательный Heap Sort
void heapSortSequential(int* arr, int n) {
    for (int i = n / 2 - 1; i >= 0; i--)       // Построение max-кучи
        heapify(arr, n, i);

    for (int i = n - 1; i >= 0; i--) {         // Извлечение элементов
        swap(arr[0], arr[i]);                   // Меняем корень с последним элементом
        heapify(arr, i, 0);                     // Heapify оставшейся части
    }
}

// ======================================================
// Параллельные алгоритмы с OpenMP
// ======================================================

// Параллельный Merge Sort
void mergeSortParallel(int* arr, int l, int r, int depth = 0) {
    if (l < r) {
        int m = l + (r - l) / 2;               // Индекс середины
        if (depth < 4) {                        // Ограничение глубины параллелизма
#pragma omp parallel sections
            {
#pragma omp section
                mergeSortParallel(arr, l, m, depth + 1);   // Левая половина в отдельной секции
#pragma omp section
                mergeSortParallel(arr, m + 1, r, depth + 1); // Правая половина в отдельной секции
            }
        } else {                                 // Если достигли максимальной глубины
            mergeSortSequential(arr, l, m);     // Сортируем последовательно
            mergeSortSequential(arr, m + 1, r);
        }
        merge(arr, l, m, r);                     // Сливаем подмассивы
    }
}

// Параллельный Quick Sort
void quickSortParallel(int* arr, int low, int high, int depth = 0) {
    if (low < high) {
        int pi = partition(arr, low, high);     // Получаем pivot
        if (depth < 4) {
#pragma omp parallel sections
            {
#pragma omp section
                quickSortParallel(arr, low, pi - 1, depth + 1);   // Левая часть
#pragma omp section
                quickSortParallel(arr, pi + 1, high, depth + 1);  // Правая часть
            }
        } else {                                 // Последовательная сортировка на глубине
            quickSortSequential(arr, low, pi - 1);
            quickSortSequential(arr, pi + 1, high);
        }
    }
}

// Параллельный Heap Sort (построение кучи параллельно)
void heapifyParallel(int* arr, int n, int i) {
    int largest = i;
    int l = 2 * i + 1;
    int r = 2 * i + 2;

    if (l < n && arr[l] > arr[largest]) largest = l;
    if (r < n && arr[r] > arr[largest]) largest = r;

    if (largest != i) {
        swap(arr[i], arr[largest]);
        heapifyParallel(arr, n, largest);
    }
}

void heapSortParallel(int* arr, int n) {
#pragma omp parallel for                   // Параллельный цикл по всем родителям
    for (int i = n / 2 - 1; i >= 0; i--)
        heapifyParallel(arr, n, i);

    for (int i = n - 1; i >= 0; i--) {     // Последовательная часть: извлечение элементов
        swap(arr[0], arr[i]);
        heapifyParallel(arr, i, 0);
    }
}

// ======================================================
// Функция тестирования всех алгоритмов
// ======================================================
void runTest(int size) {
    cout << "\n========== Размер массива: " << size << " ==========\n";

    int* originalArr = new int[size];        // Создаем исходный массив
    random_device rd;
    mt19937 gen(rd());                       // Генератор случайных чисел
    uniform_int_distribution<> dist(RAND_MIN_VAL, RAND_MAX_VAL); // Диапазон

    for (int i = 0; i < size; i++)
        originalArr[i] = dist(gen);          // Заполняем массив случайными числами

    // Lambda для последовательных алгоритмов с 3 параметрами
    auto benchmark = [&](string name, string mode, void (*sortFunc)(int*, int, int)) {
        int* tempArr = new int[size];                   // Копируем массив
        copy(originalArr, originalArr + size, tempArr);

        auto start = chrono::high_resolution_clock::now();  // Время начала
        sortFunc(tempArr, 0, size - 1);                     // Сортировка
        auto end = chrono::high_resolution_clock::now();    // Время окончания

        chrono::duration<double> elapsed = end - start;     // Длительность

        cout << "Алгоритм: " << name
             << " | Режим: " << mode
             << " | Время: " << elapsed.count() * 1000 << " мс | "
             << (isSorted(tempArr, size) ? "Массив отсортирован" : "Ошибка сортировки") << endl;

        delete[] tempArr; // Удаляем временный массив
    };

    // Lambda для Heap Sort с 2 параметрами
    auto benchmarkHeap = [&](string name, string mode, void (*sortFunc)(int*, int)) {
        int* tempArr = new int[size];
        copy(originalArr, originalArr + size, tempArr);

        auto start = chrono::high_resolution_clock::now();
        sortFunc(tempArr, size);
        auto end = chrono::high_resolution_clock::now();

        chrono::duration<double> elapsed = end - start;

        cout << "Алгоритм: " << name
             << " | Режим: " << mode
             << " | Время: " << elapsed.count() * 1000 << " мс | "
             << (isSorted(tempArr, size) ? "Массив отсортирован" : "Ошибка сортировки") << endl;

        delete[] tempArr;
    };

    // Запуск всех алгоритмов для данного размера массива
    benchmark("Merge Sort", "Последовательный", [](int* arr, int l, int r){ mergeSortSequential(arr, l, r); });
    benchmark("Quick Sort", "Последовательный", [](int* arr, int l, int r){ quickSortSequential(arr, l, r); });
    benchmarkHeap("Heap Sort", "Последовательный", heapSortSequential);

    benchmark("Merge Sort", "Параллельный", [](int* arr, int l, int r){ mergeSortParallel(arr, l, r, 0); });
    benchmark("Quick Sort", "Параллельный", [](int* arr, int l, int r){ quickSortParallel(arr, l, r, 0); });
    benchmarkHeap("Heap Sort", "Параллельный", heapSortParallel);

    delete[] originalArr; // Освобождаем исходный массив
}

// ======================================================
// Главная функция
// ======================================================
int SortingAlgorithmsSequentionalAndOpenMP() {
    omp_set_num_threads(4);                    // Устанавливаем количество потоков OpenMP
    int sizes[] = {100, 1000, 10000, 100000, 1000000}; // Размеры тестовых массивов
    for (int size : sizes) {
        runTest(size);                         // Запуск тестов
    }
    return 0;                                  // Успешное завершение программы
}
