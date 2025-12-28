#include <iostream>      // Библиотека для ввода/вывода
#include <vector>        // Библиотека для использования динамических массивов std::vector
#include <omp.h>         // Библиотека OpenMP для параллельного программирования
#include <algorithm>     // Библиотека для стандартных алгоритмов (например, swap)
#include <random>        // Библиотека для генерации случайных чисел
#include <chrono>        // Библиотека для измерения времени выполнения
#include <iomanip>       // Библиотека для форматированного вывода

using namespace std;     // Используем стандартное пространство имён
// Структура для хранения значения и индекса (для кастомного reduction)
struct Compare {
    int val;     // Значение элемента
    int index;   // Индекс элемента
};

// ===============================
// Пользовательский reduction для OpenMP
// ===============================

// Этот pragma объявляет новый reduction "minIdx" для структуры Compare
// Он позволяет безопасно находить минимальное значение и его индекс
// в параллельных циклах без использования критических секций или атомарных операций
#pragma omp declare reduction(minIdx : struct Compare : \
    omp_out = omp_in.val < omp_out.val ? omp_in : omp_out) \
    initializer (omp_priv = {2147483647, -1})

    // combiner: выражение, объединяющее приватные копии потоков
    // Если значение текущей приватной копии (omp_in.val) меньше глобального (omp_out.val)
    // то глобальный минимум заменяется на эту приватную копию
    // initializer: начальное значение приватной копии для каждого потока
    // Устанавливаем максимальное возможное целое число для val, чтобы любой элемент массива оказался меньше
    // Индекс ставим -1 как "пустой" или "неопределённый" индекс, он будет заменён реальным значением

// Параллельная сортировка выбором с использованием OpenMP и кастомного reduction
void selectionSortParallel(vector<int>& arr) {
    int n = arr.size(); // Размер массива
    for (int i = 0; i < n - 1; i++) { // Проходим по всем элементам кроме последнего
        Compare min_node = {arr[i], i}; // Инициализация текущего минимума

        // ==============================
        // Параллельный поиск минимального элемента
        // ==============================
        #pragma omp parallel for reduction(minIdx:min_node)
        for (int j = i + 1; j < n; j++) {
            // Каждый поток проверяет элементы j на наличие меньшего значения
            if (arr[j] < min_node.val) {
                // Если поток находит элемент меньше текущего минимума,
                // он обновляет свою приватную копию min_node
                min_node.val = arr[j];     // Новое минимальное значение
                min_node.index = j;        // Индекс этого элемента
            }
            // После завершения всех потоков OpenMP выполняет reduction:
            // сравнивает приватные min_node каждого потока с глобальным min_node
            // и оставляет глобальный минимум
        }

        swap(arr[i], arr[min_node.index]); // Меняем текущий элемент с найденным минимумом
    }
}

// Последовательная сортировка выбором
void selectionSortSequential(vector<int>& arr) {
    int n = arr.size(); // Размер массива
    for (int i = 0; i < n - 1; i++) {
        int min_idx = i; // Инициализация индекса минимального элемента текущим элементом
        for (int j = i + 1; j < n; j++) {
            if (arr[j] < arr[min_idx]) // Проверка на новый минимум
                min_idx = j;           // Обновление индекса минимума
        }
        swap(arr[i], arr[min_idx]);   // Обмен текущего элемента с найденным минимумом
    }
}

// Функция для тестирования сортировки на массиве заданного размера
void runTest(int size, int randMin = 1, int randMax = 1000000) {
    vector<int> dataSeq(size); // Массив для последовательной сортировки
    vector<int> dataPar(size); // Массив для параллельной сортировки

    // ==============================
    // Генерация случайных чисел
    // ==============================
    random_device rd;                   // Источник энтропии
    mt19937 gen(rd());                  // Генератор случайных чисел Mersenne Twister
    uniform_int_distribution<> dist(randMin, randMax); // Равномерное распределение

    for (int i = 0; i < size; i++) {
        dataSeq[i] = dist(gen);         // Заполнение массива случайными числами
    }
    dataPar = dataSeq;                  // Копируем массив для параллельной сортировки

    cout << "\n--- Размер массива: " << size << " ---\n";
    cout << fixed << setprecision(10);  // Форматированный вывод

    // ==============================
    // Последовательная сортировка
    // ==============================
    auto start_seq = chrono::high_resolution_clock::now(); // Время начала
    selectionSortSequential(dataSeq);                       // Последовательная сортировка
    auto end_seq = chrono::high_resolution_clock::now();    // Время окончания
    chrono::duration<double> time_seq = end_seq - start_seq; // Длительность выполнения
    cout << "Последовательная сортировка: " << time_seq.count() << " секунд\n";

    // ==============================
    // Параллельная сортировка
    // ==============================
    auto start_par = chrono::high_resolution_clock::now(); // Время начала
    selectionSortParallel(dataPar);                         // Параллельная сортировка
    auto end_par = chrono::high_resolution_clock::now();    // Время окончания
    chrono::duration<double> time_par = end_par - start_par; // Длительность выполнения
    cout << "Параллельная сортировка с OpenMP: " << time_par.count() << " секунд\n";

    // Вычисление ускорения
    cout << "Ускорение: " << time_seq.count() / time_par.count() << "x\n";
}

// Главная функция программы
int SelectionSortingAlgorithm() {
    omp_set_num_threads(4);            // Устанавливаем 4 потока OpenMP
    int sizes[] = {100, 1000, 10000, 100000};  // Размеры массивов для тестирования

    for (int size : sizes) {
        runTest(size);                  // Тестируем каждый размер массива
    }
    return 0;                           // Завершение программы
}
int main() {
        SelectionSortingAlgorithm();
        return 0;
}
