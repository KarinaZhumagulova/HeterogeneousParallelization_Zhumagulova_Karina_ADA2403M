#include <iostream>  // Библиотека для ввода-вывода (cout, endl)
#include <vector>    // Динамические массивы (контейнер vector)
#include <cmath>     // Математические функции
#include <omp.h>     // Заголовочный файл OpenMP для многопоточности
#include <random>    // Библиотека для генерации случайных чисел

using namespace std; // Использование пространства имен std для краткости

// Определение макросов для границ генерации случайных чисел
#define RAND_MIN_VAL -100
#define RAND_MAX_VAL 100

// Функция для печати первых и последних 10 элементов вектора
void print_edges(const vector<double>& data) {
    size_t n = data.size(); // Получаем размер вектора

    cout << "First 10 elements: ";
    // Цикл от 0 до 10 (или до размера вектора, если он меньше 10)
    for (size_t i = 0; i < min(n, size_t(10)); i++)
        cout << data[i] << " ";
    cout << endl;

    cout << "Last 10 elements:  ";
    // Цикл для вывода последних 10 элементов
    for (size_t i = (n > 10 ? n - 10 : 0); i < n; i++)
        cout << data[i] << " ";
    cout << endl;
}

int main() {

    // Список размеров массивов, которые мы будем тестировать
    vector<size_t> sizes = {10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000};

    // Список вариантов количества потоков для эксперимента
    vector<int> thread_counts = {1, 2, 4, 8};

    // Настройка генератора случайных чисел
    random_device rd;  // Источник энтропии
    mt19937 gen(rd()); // Инициализация генератора Вихрь Мерсенна
    uniform_int_distribution<> dist(RAND_MIN_VAL, RAND_MAX_VAL); // Равномерное распределение целых чисел

    // Внешний цикл по количеству используемых потоков
    for (int threads : thread_counts) {

        // Установка количества потоков для всех последующих параллельных областей
        omp_set_num_threads(threads);

        cout << "\n========================================\n";
        cout << "OpenMP threads: " << threads << endl;
        cout << "========================================\n";

        // Внутренний цикл по размерам векторов
        for (size_t N : sizes) {

            cout << "\nArray size: " << N << endl;

            // Выделение памяти под вектор размером N
            vector<double> data(N);

            // ----------------------------
            // Последовательная часть
            // ----------------------------
            double t_seq_start = omp_get_wtime(); // Засекаем время начала последовательного блока

            // Заполнение массива случайными числами (выполняется одним потоком)
            for (size_t i = 0; i < N; i++) {
                data[i] = static_cast<double>(dist(gen));
            }

            double t_seq_end = omp_get_wtime(); // Засекаем время окончания
            double time_sequential = t_seq_end - t_seq_start; // Вычисляем длительность

            // Печать краев заполненного массива
            print_edges(data);

            // ----------------------------
            // Параллельная часть
            // ----------------------------
            double t_par_start = omp_get_wtime(); // Засекаем время начала параллельных вычислений

            double sum = 0.0; // Переменная для накопления суммы

            // Директива для распараллеливания цикла.
            // reduction(+:sum) создает локальную копию sum для каждого потока и суммирует их в конце.
#pragma omp parallel for reduction(+:sum)
            for (size_t i = 0; i < N; i++) {
                sum += data[i]; // Каждый поток считает сумму своей части массива
            }

            double mean = sum / N; // Вычисление среднего арифметического

            double variance = 0.0; // Переменная для накопления дисперсии

            // Распараллеливание вычисления суммы квадратов отклонений
#pragma omp parallel for reduction(+:variance)
            for (size_t i = 0; i < N; i++) {
                double diff = data[i] - mean; // Разница между элементом и средним
                variance += diff * diff;     // Накопление суммы квадратов
            }

            variance /= N; // Финальное деление для получения дисперсии

            double t_par_end = omp_get_wtime(); // Конец замера времени параллельной части
            double time_parallel = t_par_end - t_par_start;

            double total_time = time_sequential + time_parallel; // Общее время работы для данного N

            // ----------------------------
            // Вывод результатов
            // ----------------------------
            cout << "Sum:      " << sum << endl;
            cout << "Mean:     " << mean << endl;
            cout << "Variance: " << variance << endl;

            cout << "Sequential time: " << time_sequential << " s\n";
            cout << "Parallel time:   " << time_parallel << " s\n";
            cout << "Total time:      " << total_time << " s\n";

            // Вычисление и вывод доли последовательной части в общем времени
            cout << "Sequential fraction: "
                 << time_sequential / total_time << endl;

            // Вычисление и вывод доли параллельной части в общем времени
            cout << "Parallel fraction:   "
                 << time_parallel / total_time << endl;

            cout << "----------------------------------------\n";
        }
    }

    return 0; // Возврат из функции
}
