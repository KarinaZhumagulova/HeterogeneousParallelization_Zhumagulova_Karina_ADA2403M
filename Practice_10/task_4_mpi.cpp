#include <mpi.h>       // Заголовочный файл MPI для параллельного программирования
#include <iostream>    // Стандартный ввод-вывод
#include <vector>      // Контейнер для динамических массивов
#include <random>      // Генераторы случайных чисел
#include <algorithm>   // Алгоритмы (min, max и т.д.)
#include <cmath>       // Математические функции

using namespace std;

// Константы для диапазона случайных чисел
#define RAND_MIN_VAL -100
#define RAND_MAX_VAL 100

// Функция для печати первых 10 элементов (используется для контроля данных)
void print_edges(const vector<double>& data) {
    size_t n = data.size();

    cout << "First 10 elements: ";
    // Вывод первых 10 элементов или меньше, если массив короче
    for (size_t i = 0; i < min<size_t>(10, n); ++i)
        cout << data[i] << " ";
    cout << endl;
}

int main(int argc, char** argv) {

    // Инициализация среды MPI
    MPI_Init(&argc, &argv);

    int rank, size;
    // Получение номера текущего процесса (ID)
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // Получение общего количества запущенных процессов
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Список размеров массивов для тестирования производительности
    vector<size_t> sizes = {
        10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000
    };

    // Настройка генератора случайных чисел
    random_device rd;
    // К зерну (seed) добавляется rank, чтобы у каждого процесса были свои уникальные числа
    mt19937 gen(rd() + rank);
    uniform_int_distribution<> dist(RAND_MIN_VAL, RAND_MAX_VAL);

    // Цикл по всем размерам массивов
    for (size_t N : sizes) {

        // Если массив меньше количества процессов, вычисления не имеют смысла
        if (N < static_cast<size_t>(size)) {
            if (rank == 0)
                cout << "Array size " << N << " is too small for " << size << " processes\n";
            continue;
        }

        // Вычисление размера локальной порции данных для каждого процесса
        size_t local_N = N / size;
        vector<double> local_data(local_N);

        // Синхронизация всех процессов перед началом замера времени
        MPI_Barrier(MPI_COMM_WORLD);
        double start_time = MPI_Wtime(); // Начало отсчета времени

        // ----------------------------
        // Генерация данных (каждый процесс заполняет свою часть)
        // ----------------------------
        for (size_t i = 0; i < local_N; i++)
            local_data[i] = static_cast<double>(dist(gen));

        // ----------------------------
        // Локальная сумма (расчет внутри каждого процесса)
        // ----------------------------
        double local_sum = 0.0;
        for (double x : local_data)
            local_sum += x;

        double global_sum = 0.0;
        // Сбор локальных сумм со всех процессов и их сложение в глобальную сумму на процессе 0
        MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        // ----------------------------
        // Среднее (вычисляется только на главном процессе)
        // ----------------------------
        double mean = 0.0;
        if (rank == 0)
            mean = global_sum / N;

        // Рассылка вычисленного среднего значения всем процессам для расчета дисперсии
        MPI_Bcast(&mean, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // ----------------------------
        // Дисперсия (локальный расчет суммы квадратов разностей)
        // ----------------------------
        double local_var = 0.0;
        for (double x : local_data) {
            double diff = x - mean;
            local_var += diff * diff;
        }

        double global_var = 0.0;
        // Сбор локальных значений дисперсии и рассылка результата всем процессам (Allreduce)
        MPI_Allreduce(&local_var, &global_var, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // Итоговый расчет дисперсии
        double variance = global_var / N;

        // Финальная синхронизация перед окончанием замера времени
        MPI_Barrier(MPI_COMM_WORLD);
        double end_time = MPI_Wtime();
        double total_time = end_time - start_time;

        // ----------------------------
        // Вывод результатов (делает только процесс с рангом 0)
        // ----------------------------
        if (rank == 0) {

            vector<double> sample;
            // Копируем первые 10 элементов локальных данных для демонстрации
            if (!local_data.empty()) {
                size_t k = min<size_t>(10, local_data.size());
                sample.assign(local_data.begin(), local_data.begin() + k);
            }

            cout << "Processes: " << size << endl;
            cout << "Array size: " << N << endl;
            print_edges(sample); // Вывод образца данных

            cout << "Sum:      " << global_sum << endl;
            cout << "Mean:     " << mean << endl;
            cout << "Variance: " << variance << endl;
            cout << "Total time: " << total_time << " s" << endl;
            cout << "----------------------------------------\n";
        }
    }

    // Завершение работы с MPI
    MPI_Finalize();
    return 0;
}

// mpic++ -O2 main.cpp task_1.cpp task_4.cpp -o Practice_10
// mpirun -np 1 ./Practice_10
// mpirun -np 2 ./Practice_10
// mpirun -np 4 ./Practice_10
// mpirun -np 8 ./Practice_10
