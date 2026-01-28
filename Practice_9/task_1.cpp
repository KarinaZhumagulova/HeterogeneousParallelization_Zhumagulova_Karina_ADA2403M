#include <mpi.h>      // Основной заголовочный файл MPI для параллельного программирования
#include <iostream>   // Для вывода в консоль (cout)
#include <vector>     // Для работы с динамическими массивами (контейнеры)
#include <cmath>      // Для математических функций (sqrt)
#include <random>     // Для генерации случайных чисел

using namespace std;

// Константы для диапазона случайных чисел
const int RAND_MIN_VAL = 0;
const int RAND_MAX_VAL = 100;

int main(int argc, char** argv) {
    // Инициализация среды MPI. Принимает аргументы командной строки.
    MPI_Init(&argc, &argv);

    int rank, size;
    // Получение уникального номера текущего процесса (от 0 до size-1)
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // Получение общего количества запущенных процессов
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Список размеров массивов, на которых мы будем проводить замеры
    vector<int> test_sizes = {10, 100, 1000, 10000, 100000, 1000000};

    // Основной цикл по разным объемам данных
    for (int N : test_sizes) {
        vector<int> data;           // Основной массив (будет заполнен только на процессе 0)
        vector<int> counts(size);   // Массив: сколько элементов отправить каждому процессу
        vector<int> displs(size);   // Массив: смещения (отступы) в исходном массиве для каждого процесса

        // Фиксация времени начала выполнения итерации
        double start_time = MPI_Wtime();

        // Блок выполняется только "главным" процессом (Master)
        if (rank == 0) {
            data.resize(N); // Резервируем память под N элементов
            random_device rd;  // Источник энтропии для генератора
            mt19937 gen(rd()); // Инициализация генератора Mersenne Twister
            uniform_int_distribution<> dist(RAND_MIN_VAL, RAND_MAX_VAL); // Определение диапазона

            // Заполнение массива случайными числами
            for (int i = 0; i < N; ++i) {
                data[i] = dist(gen);
            }

            // Логика разбиения массива на части, если N не делится на количество процессов нацело
            int rem = N % size; // Остаток от деления
            int sum = 0;        // Накопитель для расчета смещений
            for (int i = 0; i < size; ++i) {
                // Если есть остаток, первые 'rem' процессов получают на 1 элемент больше
                counts[i] = N / size + (i < rem ? 1 : 0);
                displs[i] = sum; // Указываем, с какого индекса в 'data' начинаются данные процесса i
                sum += counts[i]; // Увеличиваем смещение на размер текущего блока
            }

            // Информационный вывод данных в консоль
            cout << "Array size: " << N << endl;
            cout << "First 10 elements: ";
            for (int i = 0; i < min(10, N); ++i) cout << data[i] << " ";
            cout << endl;

            cout << "Last 10 elements: ";
            for (int i = max(0, N - 10); i < N; ++i) cout << data[i] << " ";
            cout << endl;
        }

        // Переменная для хранения размера порции данных текущего процесса
        int local_n;
        // MPI_Scatter: процесс 0 рассылает по одному числу из counts каждому процессу в его local_n
        MPI_Scatter(counts.data(), 1, MPI_INT, &local_n, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Выделение памяти под локальный кусок данных на каждом процессе
        vector<int> local_data(local_n);

        // MPI_Scatterv: распределяет разные по размеру части массива data между процессами
        // Использует counts (размеры) и displs (смещения), подготовленные на rank 0
        MPI_Scatterv(data.data(), counts.data(), displs.data(), MPI_INT,
                     local_data.data(), local_n, MPI_INT, 0, MPI_COMM_WORLD);

        // Переменные для промежуточных вычислений внутри каждого процесса
        double local_sum = 0;
        double local_sq_sum = 0;
        // Вычисление суммы и суммы квадратов элементов локального подмассива
        for (int x : local_data) {
            local_sum += x;
            local_sq_sum += (double)x * x; // Приведение к double для точности
        }

        // Переменные для итоговых результатов (имеют значение только на rank 0)
        double total_sum = 0, total_sq_sum = 0;

        // MPI_Reduce: собирает local_sum со всех процессов, суммирует их (MPI_SUM) и кладет в total_sum на rank 0
        MPI_Reduce(&local_sum, &total_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        // Аналогично собираем сумму квадратов для вычисления дисперсии
        MPI_Reduce(&local_sq_sum, &total_sq_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        // Фиксация времени окончания вычислений
        double end_time = MPI_Wtime();

        // Финальный расчет и вывод (только на главном процессе)
        if (rank == 0) {
            double mean = total_sum / N; // Среднее арифметическое
            // Формула стандартного отклонения: sqrt( E[X^2] - (E[X])^2 )
            double stddev = sqrt(total_sq_sum / N - mean * mean);
            cout << "Mean: " << mean << " | StdDev: " << stddev << endl;
            cout << "Execution time: " << end_time - start_time << " seconds." << endl;
            cout << "------------------------------------------" << endl;
        }

        // Барьерная синхронизация: все процессы ждут здесь, пока последний не закончит итерацию
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Завершение работы с MPI, очистка ресурсов
    MPI_Finalize();
    return 0;
}
