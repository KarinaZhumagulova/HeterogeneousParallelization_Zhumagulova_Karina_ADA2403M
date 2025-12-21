#include <iostream>   // Подключение библиотеки для ввода и вывода
#include <random>     // Подключение библиотеки для генерации случайных чисел
#include <chrono>     // Подключение библиотеки для измерения времени
#include <omp.h>      // Подключение библиотеки OpenMP для параллельных вычислений
#include <iomanip>    // Подключение библиотеки для форматированного вывода

using namespace std;  // Использование стандартного пространства имён

int task1() { // Функция для выполнения задания 1

    constexpr int SIZE = 50000;           // Константа размера массива
    constexpr int RAND_MIN_VAL = 1;       // Минимальное значение случайного диапазона
    constexpr int RAND_MAX_VAL = 100;     // Максимальное значение случайного диапазона

    int* arr = new int[SIZE];             // Динамическое выделение памяти под массив целых чисел

    random_device rd;                     // Источник энтропии для генератора
    mt19937 gen(rd());                 // Генератор случайных чисел Mersenne Twister
    uniform_int_distribution<> dist(RAND_MIN_VAL, RAND_MAX_VAL); // Равномерное распределение

    for (int i = 0; i < SIZE; i++) {         // Цикл заполнения массива
        arr[i] = dist(gen);               // Генерация случайного числа и запись в массив
    }

    auto start_seq = chrono::high_resolution_clock::now(); // Начало измерения времени (последовательно)

    long long sum_seq = 0;                // Переменная для хранения суммы элементов (последовательно)

    for (int i = 0; i < SIZE; i++) {      // Последовательный цикл суммирования
        sum_seq += arr[i];                // Добавление элемента массива к сумме
    }

    double avg_seq = static_cast<double>(sum_seq) / SIZE; // Вычисление среднего значения

    auto end_seq = chrono::high_resolution_clock::now();  // Конец измерения времени (последовательно)
    chrono::duration<double> time_seq = end_seq - start_seq; // Вычисление времени выполнения

    auto start_par = chrono::high_resolution_clock::now(); // Начало измерения времени (параллельно)

    long long sum_par = 0;                // Переменная для хранения суммы элементов (параллельно)

    #pragma omp parallel for reduction(+:sum_par) // Параллельный цикл с редукцией суммы
    for (int i = 0; i < SIZE; i++) {      // Параллельный цикл суммирования
        sum_par += arr[i];                // Добавление элемента массива к общей сумме
    }

    double avg_par = static_cast<double>(sum_par) / SIZE; // Вычисление среднего значения

    auto end_par = chrono::high_resolution_clock::now();  // Конец измерения времени (параллельно)
    chrono::duration<double> time_par = end_par - start_par; // Вычисление времени выполнения

    cout << fixed << setprecision(10);    // Установка формата вывода с 10 знаками после запятой
    cout << "Последовательное вычисление." << endl;
    cout << "Среднее значение массива: " << avg_seq << endl; // Вывод среднего значения (SEQ)
    cout << "Время выполнения: " << time_seq.count() << " секунды\n" << endl; // Вывод времени (SEQ)

    cout << "Параллельное вычисление с использованием OpenMP." << endl;
    cout << "Среднее значение массива: " << avg_par << endl; // Вывод среднего значения (OMP)
    cout << "Время выполнения: " << time_par.count() << " секунды\n" << endl; // Вывод времени (OMP)

    delete[] arr;                          // Освобождение динамически выделенной памяти
    cout << "Динамическая память успешно освобождена." << endl; // Сообщение об освобождении памяти

    return 0;                              // Завершение функции
}

int main() {          // Главная функция программы
    task1();          // Вызов функции задания 1
    return 0;         // Завершение работы программы
}
