#include <iostream>      // Ввод и вывод данных
#include <vector>        // Контейнер vector
#include <random>        // Генерация случайных чисел
#include <chrono>        // Измерение времени выполнения
#include <iomanip>       // Форматированный вывод (setprecision)
#include <omp.h>         // Библиотека OpenMP

using namespace std;     // Использование пространства имён std

// --- Настройки диапазона случайных чисел ---
constexpr int RAND_MIN_VAL = 1;        // <-- менять здесь минимум диапазона
constexpr int RAND_MAX_VAL = 999999;  // <-- менять здесь максимум диапазона

// Функция для работы с массивом
void arrayTask() {

    int N;                                      // Размер массива
    cout << "Enter N: ";                        // Запрос размера массива
    cin >> N;                                   // Ввод размера массива

    vector<int> arr(N);                         // Создание массива из N элементов

    // --- Генерация случайных чисел ---
    random_device rd;                           // Источник энтропии
    mt19937 gen(rd());                       // Генератор случайных чисел
    uniform_int_distribution<> dist(RAND_MIN_VAL, RAND_MAX_VAL); // Диапазон

    for (int i = 0; i < N; ++i) {                  // Цикл заполнения массива
        arr[i] = dist(gen);                     // Присваивание случайного числа
    }

    // --- Вывод массива ---
    // cout << "Array: ";                          // Заголовок вывода массива
    for (int x : arr) {                         // Перебор элементов массива
        x; //cout << x << " ";                       // Вывод элемента
    }
    //cout << "\n\n";                             // Перевод строки

    // ================= ПОСЛЕДОВАТЕЛЬНЫЙ ВАРИАНТ =================

    auto start_seq = chrono::high_resolution_clock::now(); // Начало измерения времени

    int min_seq = arr[0];                       // Начальное минимальное значение
    int max_seq = arr[0];                       // Начальное максимальное значение

    for (int i = 1; i < N; ++i) {               // Последовательный обход массива
        if (arr[i] < min_seq) min_seq = arr[i]; // Обновление минимума
        if (arr[i] > max_seq) max_seq = arr[i]; // Обновление максимума
    }

    auto end_seq = chrono::high_resolution_clock::now();   // Конец измерения времени
    chrono::duration<double> time_seq = end_seq - start_seq; // Время выполнения

    // ================= ПАРАЛЛЕЛЬНЫЙ ВАРИАНТ (OpenMP) =================

    auto start_par = chrono::high_resolution_clock::now(); // Начало измерения времени

    int min_par = RAND_MAX_VAL;               // Начальное значение минимума
    int max_par = RAND_MIN_VAL;               // Начальное значение максимума

    #pragma omp parallel for reduction(min:min_par) reduction(max:max_par)
    for (int i = 0; i < N; ++i) {               // Параллельный цикл OpenMP
        if (arr[i] < min_par) min_par = arr[i]; // Локальный минимум
        if (arr[i] > max_par) max_par = arr[i]; // Локальный максимум
    }

    auto end_par = chrono::high_resolution_clock::now();   // Конец измерения времени
    chrono::duration<double> time_par = end_par - start_par; // Время выполнения

    // --- Формат вывода времени ---
    cout << fixed << setprecision(11);          // Формат 0.00000000000

    // --- Вывод результатов ---
    cout << "Sequential version:\n";            // Заголовок последовательного варианта
    cout << "Min = " << min_seq                 // Вывод минимума
         << ", Max = " << max_seq << "\n";      // Вывод максимума
    cout << "Time: " << time_seq.count()        // Вывод времени
         << " seconds\n\n";

    cout << "Parallel version (OpenMP):\n";     // Заголовок параллельного варианта
    cout << "Min = " << min_par                 // Вывод минимума
         << ", Max = " << max_par << "\n";      // Вывод максимума
    cout << "Time: " << time_par.count()        // Вывод времени
         << " seconds\n";
}
int main() {
    arrayTask(); // Вызов функции для выполнения Задания № 1
    return 0;
}
