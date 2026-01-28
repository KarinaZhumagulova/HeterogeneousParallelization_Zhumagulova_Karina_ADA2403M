#include <mpi.h>      // Подключение библиотеки MPI для параллельных вычислений
#include <iostream>   // Подключение библиотеки ввода-вывода
#include <vector>     // Подключение контейнера динамических массивов
#include <iomanip>    // Для форматированного вывода (setw, setprecision)
#include <cstdlib>    // Для функций rand() и atoi()
#include <cmath>      // Для математических функций

using namespace std;

int main(int argc, char** argv) {
    // Инициализация окружения MPI
    MPI_Init(&argc, &argv);

    int rank, size;
    // Получение номера текущего процесса (rank)
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // Получение общего количества процессов (size)
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 6; // Размерность матрицы по умолчанию (6x6)
    // Если передан аргумент командной строки, используем его как размер матрицы
    if (argc > 1) N = atoi(argv[1]);

    vector<vector<double>> A; // Двумерный вектор для хранения матрицы на процессе 0
    vector<double> b;         // Вектор для хранения правой части (свободных членов)

    if (rank == 0) {
        // Резервируем память под матрицу и вектор b на главном процессе
        A.resize(N, vector<double>(N));
        b.resize(N);
        srand(0); // Инициализация генератора случайных чисел для воспроизводимости
        for (int i = 0; i < N; ++i) {
            b[i] = rand() % 10 + 1; // Заполнение вектора случайными числами от 1 до 10
            for (int j = 0; j < N; ++j) {
                A[i][j] = rand() % 10 + 1; // Заполнение матрицы случайными числами
            }
        }

        // Вывод исходной матрицы и вектора b в консоль
        cout << "Original system:" << endl;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) cout << setw(5) << A[i][j] << " ";
            cout << "| " << setw(5) << b[i] << endl;
        }
        cout << "-----------------------------------" << endl;
    }

    // ---------------- Раздаём строки между процессами ----------------
    // sendcounts - сколько строк получит каждый процесс, displs - смещения в массиве
    vector<int> sendcounts(size), displs(size);
    int rows_per_proc = N / size; // Базовое количество строк на процесс
    int remainder = N % size;     // Остаток строк, если N не делится на size

    // Распределение нагрузки: первые 'remainder' процессов получают на одну строку больше
    for (int i = 0; i < size; ++i) {
        sendcounts[i] = rows_per_proc + (i < remainder ? 1 : 0);
        displs[i] = (i == 0 ? 0 : displs[i-1] + sendcounts[i-1]);
    }

    int local_n = sendcounts[rank]; // Количество строк, которое обработает текущий процесс
    vector<double> local_A(local_n * N); // Локальный буфер для строк матрицы (в плоском виде)
    vector<double> local_b(local_n);     // Локальный буфер для соответствующих элементов b

    // Для передачи через MPI_Scatterv матрицу нужно "развернуть" в одномерный массив
    vector<double> A_flat;
    if (rank == 0) {
        A_flat.resize(N * N);
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                A_flat[i*N + j] = A[i][j]; // Копирование из 2D вектора в 1D массив
    }

    // Подготовка массивов смещений и количеств специально для данных матрицы A
    vector<int> sendcounts_A(size), displs_A(size);
    for (int i = 0; i < size; ++i) {
        sendcounts_A[i] = sendcounts[i] * N; // Количество элементов (строки * столбцы)
        displs_A[i] = displs[i] * N;         // Смещение в общем массиве
    }

    // Рассылка частей матрицы A всем процессам
    MPI_Scatterv(A_flat.data(), sendcounts_A.data(), displs_A.data(), MPI_DOUBLE,
                 local_A.data(), local_n * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Рассылка соответствующих частей вектора b всем процессам
    MPI_Scatterv(b.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                 local_b.data(), local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // ------------------- Прямой ход метода Гаусса -------------------
    // pivot_row хранит текущую ведущую строку и соответствующий элемент b (N+1 элемент)
    vector<double> pivot_row(N+1);

    for (int k = 0; k < N; ++k) {
        // Поиск процесса, которому принадлежит k-я (ведущая) строка
        int owner = 0;
        while (k >= displs[owner] + sendcounts[owner]) owner++;
        int local_k = k - displs[owner]; // Индекс строки внутри процесса-владельца

        // Если текущий процесс — владелец k-й строки, он готовит данные для рассылки
        if (rank == owner) {
            for (int j = 0; j < N; ++j) pivot_row[j] = local_A[local_k * N + j];
            pivot_row[N] = local_b[local_k];
        }

        // Рассылка ведущей строки всем процессам (Broadcast)
        MPI_Bcast(pivot_row.data(), N+1, MPI_DOUBLE, owner, MPI_COMM_WORLD);

        // Каждый процесс обновляет только те свои строки, которые находятся ниже k-й
        for (int i = 0; i < local_n; ++i) {
            int global_row = displs[rank] + i; // Преобразование локального индекса в глобальный
            if (global_row <= k) continue;     // Пропускаем строки выше текущей ведущей

            // Вычисление коэффициента для обнуления элемента в столбце k
            double factor = local_A[i*N + k] / pivot_row[k];
            // Вычитание ведущей строки из текущей
            for (int j = k; j < N; ++j)
                local_A[i*N + j] -= factor * pivot_row[j];
            local_b[i] -= factor * pivot_row[N]; // Обновление вектора b
        }
    }

    // ------------------- Сбор результатов -------------------
    // Сборка измененной матрицы (верхнетреугольной) обратно на процесс 0
    MPI_Gatherv(local_A.data(), local_n * N, MPI_DOUBLE,
                A_flat.data(), sendcounts_A.data(), displs_A.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Сборка обновленного вектора b на процесс 0
    MPI_Gatherv(local_b.data(), local_n, MPI_DOUBLE,
                b.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // ------------------- Обратный ход на rank=0 -------------------
    if (rank == 0) {
        vector<double> x(N); // Вектор решений
        // Вычисление неизвестных в обратном порядке (от x_n до x_1)
        for (int i = N-1; i >= 0; --i) {
            x[i] = b[i]; // Начинаем со свободного члена
            // Вычитаем уже найденные значения x
            for (int j = i+1; j < N; ++j)
                x[i] -= A_flat[i*N + j] * x[j];
            // Делим на коэффициент при x_i
            x[i] /= A_flat[i*N + i];
        }

        // Вывод итогового вектора решения
        cout << "Solution vector x:" << endl;
        for (int i = 0; i < N; ++i) cout << fixed << setprecision(3) << x[i] << " ";
        cout << endl;
    }

    // Завершение работы MPI
    MPI_Finalize();
    return 0;
}
