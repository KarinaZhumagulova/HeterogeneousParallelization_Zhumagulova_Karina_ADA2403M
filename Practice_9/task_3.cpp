#include <mpi.h>      // Подключение библиотеки MPI для параллельных вычислений
#include <iostream>   // Потоки ввода-вывода
#include <vector>     // Контейнер vector
#include <iomanip>    // Манипуляторы вывода для красивого форматирования таблиц
#include <cstdlib>    // Стандартные функции (rand, srand, atoi)
#include <cmath>      // Математические функции

using namespace std;

const double INF = 1e9; // Определение бесконечности для обозначения отсутствия ребра

int main(int argc, char** argv) {
    // Инициализация среды MPI
    MPI_Init(&argc, &argv);

    int rank, size;
    // Получение идентификатора текущего процесса
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // Получение общего количества запущенных процессов
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Начало замера времени выполнения всей программы
    double start_time = MPI_Wtime();

    // Список размеров графов, которые будут протестированы последовательно
    vector<int> test_sizes = {3, 5, 7, 10, 50, 100, 150, 200, 500, 1000};

    for (int N : test_sizes) {
        vector<vector<double>> G; // Двумерный вектор для хранения графа на процессе 0
        if (rank == 0) {
            // Резервирование памяти под матрицу смежности NxN и заполнение INF
            G.resize(N, vector<double>(N, INF));
            srand(0); // Инициализация генератора для повторяемости результатов
            for (int i = 0; i < N; ++i) {
                G[i][i] = 0; // Расстояние от вершины до самой себя всегда 0
                for (int j = 0; j < N; ++j) {
                    if (i != j) {
                        // С вероятностью 20% ребра нет (INF), иначе случайный вес от 1 до 10
                        G[i][j] = (rand() % 5 == 0) ? INF : rand() % 10 + 1;
                    }
                }
            }

            // Вывод исходной матрицы в консоль, если граф небольшой (до 10 вершин)
            if (N < 11) {
                cout << "Original adjacency matrix:" << endl;
                for (int i = 0; i < N; ++i) {
                    for (int j = 0; j < N; ++j)
                        if (G[i][j] >= INF) cout << setw(5) << "INF";
                        else cout << setw(5) << G[i][j];
                    cout << endl;
                }
                cout << "-----------------------------------" << endl;
            }
        }

        // ---------------- Разделение строк между процессами ----------------
        vector<int> sendcounts(size), displs(size);
        int rows_per_proc = N / size; // Сколько строк гарантированно получает каждый процесс
        int remainder = N % size;     // Лишние строки для распределения
        for (int i = 0; i < size; ++i) {
            // Распределяем остаток между первыми процессами
            sendcounts[i] = rows_per_proc + (i < remainder ? 1 : 0);
            // Вычисляем индекс начала блока данных для каждого процесса
            displs[i] = (i == 0 ? 0 : displs[i-1] + sendcounts[i-1]);
        }

        int local_n = sendcounts[rank];      // Количество строк у текущего процесса
        vector<double> local_rows(local_n * N); // Локальный буфер для хранения строк матрицы

        // Подготовка одномерного массива (линеаризация) для передачи через MPI
        vector<double> G_flat;
        if (rank == 0) {
            G_flat.resize(N*N);
            for (int i = 0; i < N; ++i)
                for (int j = 0; j < N; ++j)
                    G_flat[i*N + j] = G[i][j];
        }

        // Расчет смещений и размеров для линеаризованной матрицы (умножаем на N)
        vector<int> sendcounts_flat(size), displs_flat(size);
        for (int i = 0; i < size; ++i) {
            sendcounts_flat[i] = sendcounts[i] * N;
            displs_flat[i] = displs[i] * N;
        }

        // Рассылка строк матрицы всем процессам
        MPI_Scatterv(G_flat.data(), sendcounts_flat.data(), displs_flat.data(), MPI_DOUBLE,
                     local_rows.data(), local_n * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // ---------------- Параллельный алгоритм Флойда-Уоршелла ----------------
        // all_rows используется для хранения всей матрицы на каждой итерации (синхронизация)
        vector<double> all_rows(N*N);



        for (int k = 0; k < N; ++k) {
            // Поиск процесса, владеющего k-й строкой на текущем шаге
            int owner = 0;
            while (k >= displs[owner] + sendcounts[owner]) owner++;
            int local_k = k - displs[owner]; // Локальный индекс k-й строки у владельца

            vector<double> k_row(N); // Буфер для хранения k-й строки
            if (rank == owner) {
                // Владелец копирует k-ю строку в буфер
                for (int j = 0; j < N; ++j) k_row[j] = local_rows[local_k*N + j];
            }

            // Рассылка k-й строки всем процессам для выполнения шага релаксации
            MPI_Bcast(k_row.data(), N, MPI_DOUBLE, owner, MPI_COMM_WORLD);

            // Релаксация: каждый процесс обновляет свои локальные строки
            for (int i = 0; i < local_n; ++i) {
                for (int j = 0; j < N; ++j) {
                    // Формула: d[i][j] = min(d[i][j], d[i][k] + d[k][j])
                    double new_dist = local_rows[i*N + k] + k_row[j];
                    if (new_dist < local_rows[i*N + j]) local_rows[i*N + j] = new_dist;
                }
            }

            // Синхронизация: сборка обновленных строк со всех процессов обратно во все процессы
            // Это необходимо, так как на следующем шаге k+1 любая строка может стать ведущей
            MPI_Allgatherv(local_rows.data(), local_n*N, MPI_DOUBLE,
                           all_rows.data(), sendcounts_flat.data(), displs_flat.data(), MPI_DOUBLE,
                           MPI_COMM_WORLD);

            // Копирование собранных данных обратно в локальные строки для консистентности
            for (int i = 0; i < local_n; ++i)
                for (int j = 0; j < N; ++j)
                    local_rows[i*N + j] = all_rows[(displs[rank]+i)*N + j];
        }

        // Сбор итоговой матрицы на процессе 0
        MPI_Gatherv(local_rows.data(), local_n*N, MPI_DOUBLE,
                    G_flat.data(), sendcounts_flat.data(), displs_flat.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        if (rank == 0) {
            // Вывод результата для маленьких матриц
            if (N < 11) {
                cout << "Shortest path matrix:" << endl;
                for (int i = 0; i < N; ++i) {
                    for (int j = 0; j < N; ++j)
                        if (G_flat[i*N + j] >= INF) cout << setw(5) << "INF";
                        else cout << setw(5) << G_flat[i*N + j];
                    cout << endl;
                }
                cout << "-----------------------------------" << endl;
            }

            // Вывод времени выполнения для текущего размера графа N
            double end_time = MPI_Wtime();
            cout << "Graph size: " << N << " | Execution time: "
                 << end_time - start_time << " seconds." << endl;
        }

        // Ожидание завершения всех процессов перед переходом к следующему тесту N
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Завершение работы с MPI
    MPI_Finalize();
    return 0;
}
