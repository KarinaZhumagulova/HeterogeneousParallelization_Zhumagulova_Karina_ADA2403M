#include <iostream>     // Подключаем библиотеку для ввода и вывода (cout, endl)
#include <vector>       // Подключаем контейнер vector (в коде напрямую не используется, но допустимо)
#include <random>       // Подключаем генерацию случайных чисел
#include <chrono>       // Подключаем библиотеку для измерения времени выполнения
#include <omp.h>        // Подключаем библиотеку OpenMP для параллельных вычислений
#include <algorithm>    // Подключаем алгоритмы standard library (swap, copy)

using namespace std;    // Используем стандартное пространство имён std

// ============================================================
// Константы для диапазона генерации случайных чисел
// ============================================================

constexpr int RAND_MIN_VAL = -1000;   // Минимальное значение случайного числа
constexpr int RAND_MAX_VAL = 1000;    // Максимальное значение случайного числа

// ============================================================
// Функция проверки: отсортирован ли массив
// ============================================================

bool isSortedTask3(int* arr, int size) {     // Функция принимает указатель на массив и его размер
    for (int i = 0; i < size - 1; i++) {     // Проходим по всем элементам массива
        if (arr[i] > arr[i + 1]) {           // Если текущий элемент больше следующего
            return false;                    // Массив не отсортирован
        }
    }
    return true;                             // Если нарушений нет — массив отсортирован
}

// ============================================================
// ПОСЛЕДОВАТЕЛЬНЫЕ АЛГОРИТМЫ СОРТИРОВКИ
// ============================================================

// Последовательная пузырьковая сортировка
void bubbleSortSequentialTask3(int* arr, int n) { // Функция сортировки пузырьком
    for (int i = 0; i < n - 1; i++) {              // Внешний цикл проходов
        for (int j = 0; j < n - i - 1; j++) {      // Внутренний цикл сравнений
            if (arr[j] > arr[j + 1]) {             // Если элементы стоят неправильно
                swap(arr[j], arr[j + 1]);          // Меняем их местами
            }
        }
    }
}

// Последовательная сортировка выбором
void selectionSortSequentialTask3(int* arr, int n) { // Функция сортировки выбором
    for (int i = 0; i < n - 1; i++) {                 // Проходим по массиву
        int min_idx = i;                              // Считаем текущий элемент минимальным
        for (int j = i + 1; j < n; j++) {             // Ищем минимальный элемент справа
            if (arr[j] < arr[min_idx]) {              // Если найден меньший элемент
                min_idx = j;                          // Обновляем индекс минимума
            }
        }
        swap(arr[i], arr[min_idx]);                   // Меняем минимальный элемент с текущим
    }
}

// Последовательная сортировка вставками
void insertionSortSequentialTask3(int* arr, int n) { // Функция сортировки вставками
    for (int i = 1; i < n; i++) {                     // Начинаем со второго элемента
        int key = arr[i];                             // Запоминаем текущий элемент
        int j = i - 1;                                // Индекс предыдущего элемента
        while (j >= 0 && arr[j] > key) {              // Сдвигаем элементы вправо
            arr[j + 1] = arr[j];                      // Копируем элемент вправо
            j--;                                      // Переходим влево
        }
        arr[j + 1] = key;                             // Вставляем элемент на нужное место
    }
}

// ============================================================
// ПАРАЛЛЕЛЬНЫЕ АЛГОРИТМЫ СОРТИРОВКИ (OpenMP)
// ============================================================

// Параллельная пузырьковая сортировка (Odd-Even Sort)
void parallelBubbleSortTask3(int* arr, int n) {       // Функция параллельной сортировки
    for (int i = 0; i < n; i++) {                      // Количество фаз сортировки
        if (i % 2 == 0) {                              // Чётная фаза
#pragma omp parallel for                               // Параллелим цикл
            for (int j = 0; j < n - 1; j += 2) {       // Обрабатываем чётные пары
                if (arr[j] > arr[j + 1]) {             // Сравниваем элементы
                    swap(arr[j], arr[j + 1]);          // Меняем их местами
                }
            }
        } else {                                       // Нечётная фаза
#pragma omp parallel for                               // Параллелим цикл
            for (int j = 1; j < n - 1; j += 2) {       // Обрабатываем нечётные пары
                if (arr[j] > arr[j + 1]) {             // Сравнение элементов
                    swap(arr[j], arr[j + 1]);          // Обмен элементов
                }
            }
        }
    }
}

// Параллельная сортировка выбором
void parallelSelectionSortTask3(int* arr, int n) {    // Функция параллельной сортировки выбором
    for (int i = 0; i < n - 1; i++) {                  // Внешний цикл
        int min_idx = i;                               // Индекс минимума
        int min_val = arr[i];                          // Значение минимума

#pragma omp parallel                                   // Начало параллельной области
        {
            int local_min_idx = min_idx;               // Локальный индекс минимума
            int local_min_val = min_val;               // Локальное значение минимума

#pragma omp for nowait                                 // Параллельный поиск минимума
            for (int j = i + 1; j < n; j++) {          // Проход по массиву
                if (arr[j] < local_min_val) {          // Если найден меньший элемент
                    local_min_val = arr[j];            // Обновляем локальный минимум
                    local_min_idx = j;                 // Обновляем индекс
                }
            }

#pragma omp critical                                   // Критическая секция
            {
                if (local_min_val < min_val) {         // Сравнение локального минимума
                    min_val = local_min_val;           // Обновляем глобальный минимум
                    min_idx = local_min_idx;           // Обновляем индекс
                }
            }
        }
        swap(arr[i], arr[min_idx]);                    // Обмен элементов
    }
}

// Параллельная сортировка вставками (Shell Sort)
void parallelInsertionSortTask3(int* arr, int n) {    // Функция Shell-сортировки
    for (int gap = n / 2; gap > 0; gap /= 2) {         // Последовательность шагов
#pragma omp parallel for                               // Параллелим внешний цикл
        for (int i = 0; i < gap; i++) {                // Каждая группа обрабатывается отдельно
            for (int j = i + gap; j < n; j += gap) {  // Сортировка вставками внутри группы
                int temp = arr[j];                     // Сохраняем элемент
                int k = j;                             // Индекс перемещения
                while (k >= gap && arr[k - gap] > temp) {
                    arr[k] = arr[k - gap];             // Сдвиг элемента
                    k -= gap;                          // Переход назад
                }
                arr[k] = temp;                         // Вставка элемента
            }
        }
    }
}
// Сортировка вставкой (ограниченный параллелизм)
void insertionSortParallelTask3(int* arr, int size) {   // Функция сортировки вставками
    for (int i = 1; i < size; ++i) {                    // Перебор элементов массива
        int key = arr[i];                               // Текущий вставляемый элемент
        int j = i - 1;                                  // Индекс предыдущего элемента

        // Сдвиг элементов вправо, пока не найдено место для вставки
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];                        // Сдвиг элемента
            j--;                                        // Переход к предыдущему элементу
        }

        arr[j + 1] = key;                               // Вставка элемента на нужное место
    }
}
// ============================================================
// Функция тестирования сортировок
// ============================================================

void runTestTask3(int size) {                           // Функция тестирования

    cout << "\n========== Размер массива: " << size << " ==========\n";

    int* originalArr = new int[size];                  // Выделяем память под массив

    random_device rd;                                  // Источник энтропии
    mt19937 gen(rd());                                 // Генератор случайных чисел
    uniform_int_distribution<> dist(RAND_MIN_VAL, RAND_MAX_VAL); // Диапазон значений

    for (int i = 0; i < size; i++)                     // Заполняем массив
        originalArr[i] = dist(gen);

    auto benchmark = [&](string name, string mode,
                         void (*sortFunc)(int*, int)) {

        int* tempArr = new int[size];                  // Копия массива
        copy(originalArr, originalArr + size, tempArr);// Копируем данные

        auto start = chrono::high_resolution_clock::now(); // Начало измерения времени
        sortFunc(tempArr, size);                       // Вызов сортировки
        auto end = chrono::high_resolution_clock::now();   // Конец измерения времени

        chrono::duration<double> elapsed = end - start;// Вычисление времени

        cout << "Алгоритм: " << name
             << " | Режим: " << mode
             << " | Время: " << elapsed.count() * 1000 << " мс | ";

        cout << (isSortedTask3(tempArr, size)
                ? "Массив отсортирован корректно"
                : "Ошибка сортировки!") << endl;

        delete[] tempArr;                               // Освобождение памяти
    };

    benchmark("Пузырьковая сортировка", "Последовательный", bubbleSortSequentialTask3);
    benchmark("Сортировка выбором", "Последовательный", selectionSortSequentialTask3);
    benchmark("Сортировка вставками", "Последовательный", insertionSortSequentialTask3);

    benchmark("Пузырьковая (Odd-Even)", "Параллельный", parallelBubbleSortTask3);
    benchmark("Сортировка выбором", "Параллельный", parallelSelectionSortTask3);
    benchmark("Shell-сортировка", "Параллельный", parallelInsertionSortTask3);
    benchmark("Сортировка вставкой (ограниченный параллелизм)", "Параллельный", insertionSortParallelTask3);

    delete[] originalArr;                               // Освобождаем исходный массив
}

// ============================================================
// Главная функция
// ============================================================

int SortingAlgorithmsSequentionalAndOpenMP() {          // Точка входа

    omp_set_num_threads(4);                             // Устанавливаем 4 потока OpenMP

    int sizes[] = {10, 100, 1000, 10000, 100000};       // Размеры массивов

    for (int size : sizes) {                            // Перебор размеров
        runTestTask3(size);                             // Запуск тестов
    }

    return 0;                                           // Успешное завершение программы
}
int main() {
    SortingAlgorithmsSequentionalAndOpenMP();
    return 0;
}
