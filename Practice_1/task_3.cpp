#include <iostream>     // Ввод и вывод данных
#include <cstdlib>      // для rand() и srand()
#include <ctime>        // для time()
#include <chrono>       // для измерения времени
#ifdef _OPENMP
#include <omp.h>        // для OpenMP
#endif

using namespace std;    // Использование пространства имён std

// ================================================================
// Функция подсчёта среднего (последовательный вариант)
// ================================================================
double computeAverageSequential(int* arr, int size) {
    long long sum = 0;                 // переменная для суммы элементов
    for (int i = 0; i < size; ++i) {
        // проходим по всем элементам массива
        sum += arr[i];                 // добавляем значение элемента к сумме
    }
    return static_cast<double>(sum) / size; // вычисляем среднее
}

// ================================================================
// Функция подсчёта среднего (параллельный вариант OpenMP)
// ================================================================
double computeAverageParallel(int* arr, int size) {
    long long sum = 0;                 // переменная для суммы элементов

#ifdef _OPENMP
    #pragma omp parallel for reduction(+:sum) // параллельный цикл с суммированием
#endif
    for (int i = 0; i < size; ++i) {   // проходим по всем элементам массива
        sum += arr[i];                 // добавляем значение элемента к сумме
    }

    return static_cast<double>(sum) / size; // вычисляем среднее
}
// ================================================================
// Основная функция задания
// ================================================================
void DynamicMemoryTask() {
    cout << "====================================================\n";
    cout << "Часть 3: Динамическая память и указатели\n";
    cout << "Параллельный подсчёт среднего значения используя OpenMP\n";
    cout << "====================================================\n";

    int N;                                // переменная для размера массива
    cout << "Enter array size N: ";       // просим пользователя ввести размер
    cin >> N;                             // считываем значение N с клавиатуры

    cout << "Array size: " << N;// << "\n";

    // Создание динамического массива
    int* arr = new int[N];                // выделяем память под массив из N элементов

    // Заполнение массива случайными числами
    srand(static_cast<unsigned int>(time(0))); // инициализация генератора случайных чисел
    for (int i = 0; i < N; ++i) {        // проходим по всем элементам массива
        arr[i] = rand() % 10000;          // присваиваем случайное значение от 0 до 9999
    }
    //cout << "Array filled with random numbers (0-999)\n";

    // Печать массива
    //cout << "Array elements: ";
    for (int i = 0; i < N; ++i) {        // проходим по всем элементам массива
        arr[i]; //cout << arr[i] << " ";           // выводим элемент на экран
    }
    //cout << "\n";

    // -----------------------------
    // Последовательный подсчёт среднего
    // -----------------------------
    auto t1 = chrono::high_resolution_clock::now(); // фиксируем время начала
    double avgSeq = computeAverageSequential(arr, N); // вычисляем среднее последовательным методом
    auto t2 = chrono::high_resolution_clock::now(); // фиксируем время окончания
    chrono::duration<double, milli> durSeq = t2 - t1; // вычисляем длительность в миллисекундах

    cout << "\n[SEQUENTIAL]\n";
    cout << "Average: " << avgSeq << "\n";         // вывод среднего
    cout << "Time: " << durSeq.count() << " ms\n"; // вывод времени

    // -----------------------------
    // Параллельный подсчёт среднего
    // -----------------------------
    auto t3 = chrono::high_resolution_clock::now(); // фиксируем время начала
    double avgPar = computeAverageParallel(arr, N); // вычисляем среднее параллельным методом
    auto t4 = chrono::high_resolution_clock::now(); // фиксируем время окончания
    chrono::duration<double, milli> durPar = t4 - t3; // вычисляем длительность в миллисекундах

    cout << "\n[PARALLEL]\n";
    cout << "Average: " << avgPar << "\n";         // вывод среднего
    cout << "Time: " << durPar.count() << " ms\n"; // вывод времени

#ifdef _OPENMP
    double speedup = durSeq.count() / durPar.count(); // вычисляем ускорение
    cout << "Speedup: " << speedup << "x\n";         // вывод ускорения
#endif

    // -----------------------------
    // Освобождение памяти
    // -----------------------------
    delete[] arr;                           // освобождаем динамически выделенную память
    cout << "\nDynamic memory freed.\n";
    cout << "Task completed successfully.\n";
}
int main() {
    DynamicMemoryTask();
    return 0;
}
