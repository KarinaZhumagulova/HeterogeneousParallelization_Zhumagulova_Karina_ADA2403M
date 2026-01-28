#include <mpi.h>      // Подключение библиотеки интерфейса передачи сообщений (MPI)
#include <iostream>   // Подключение стандартного потока ввода-вывода
#include <vector>     // Подключение контейнера динамических массивов STL
#include <iomanip>    // Подключение манипуляторов для форматирования вывода (setw, setprecision)
#include <cstdlib>    // Подключение стандартных функций (atoi, rand, srand)
#include <cmath>      // Подключение математических функций

using namespace std;

int main(int argc, char** argv) {
    // Инициализация среды MPI
    MPI_Init(&argc, &argv);

    int rank, size;
    // Получение уникального номера текущего процесса (от 0 до size-1)
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // Получение общего количества запущенных процессов
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Фиксация времени начала работы программы на всех процессах
    double start_time = MPI_Wtime();

    int N = 6; // Размерность матрицы по умолчанию
    // Если передан аргумент в командной строке, преобразуем его в целое число N
    if (argc > 1) N = atoi(argv[1]);

    vector<vector<double>> A; // Матрица коэффициентов (используется только на rank 0)
    vector<double> b;         // Вектор свободных членов (используется только на rank 0)

    if (rank == 0) {
        // Выделение памяти под матрицу и вектор на главном процессе
        A.resize(N, vector<double>(N));
        b.resize(N);
        srand(0); // Инициализация генератора случайных чисел фиксированным зерном
        for (int i = 0; i < N; ++i) {
            b[i] = rand() % 10 + 1; // Случайное число от 1 до 10 для вектора b
            for (int j = 0; j < N; ++j) {
                A[i][j] = rand() % 10 + 1; // Случайное число от 1 до 10 для матрицы A
            }
        }

        // Вывод исходной системы уравнений для наглядности
        cout << "Original system:" << endl;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) cout << setw(5) << A[i][j] << " ";
            cout << "| " << setw(5) << b[i] << endl;
        }
        cout << "-----------------------------------" << endl;
    }

    // Подготовка массивов для распределения данных между процессами
    vector<int> sendcounts(size), displs(size);
    int rows_per_proc = N / size; // Базовое количество строк на один процесс
    int remainder = N % size;     // Остаток строк, если N не делится на size нацело

    // Расчет количества строк и смещений для каждого процесса
    for (int i = 0; i < size; ++i) {
        // Если есть остаток, первые процессы получают на одну строку больше (балансировка)
        sendcounts[i] = rows_per_proc + (i < remainder ? 1 : 0);
        // Смещение — это сумма длин всех предыдущих блоков
        displs[i] = (i == 0 ? 0 : displs[i-1] + sendcounts[i-1]);
    }

    int local_n = sendcounts[rank];      // Сколько строк досталось текущему процессу
    vector<double> local_A(local_n * N); // Локальный массив для хранения части матрицы A
    vector<double> local_b(local_n);     // Локальный вектор для хранения части вектора b

    vector<double> A_flat; // "Плоский" массив для матрицы A (одномерный для передачи в MPI)
    if (rank == 0) {
        A_flat.resize(N * N);
        // Преобразование двумерного вектора A в одномерный массив A_flat
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                A_flat[i*N + j] = A[i][j];
    }

    // Подготовка параметров для Scatterv специально для данных матрицы A (умножаем на N)
    vector<int> sendcounts_A(size), displs_A(size);
    for (int i = 0; i < size; ++i) {
        sendcounts_A[i] = sendcounts[i] * N; // Количество элементов в блоке строк
        displs_A[i] = displs[i] * N;         // Смещение в элементах
    }

    // Рассылка частей плоской матрицы A всем процессам в их локальные local_A
    MPI_Scatterv(A_flat.data(), sendcounts_A.data(), displs_A.data(), MPI_DOUBLE,
                 local_A.data(), local_n * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Рассылка соответствующих частей вектора b всем процессам в их local_b
    MPI_Scatterv(b.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                 local_b.data(), local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Вспомогательный вектор для хранения "ведущей" строки (N элементов строки + 1 элемент b)
    vector<double> pivot_row(N+1);

    // Прямой ход метода Гаусса: приведение матрицы к верхнетреугольному виду
    for (int k = 0; k < N; ++k) {
        // Определяем, какой процесс владеет текущей k-й строкой (ведущей)
        int owner = 0;
        while (k >= displs[owner] + sendcounts[owner]) owner++;
        int local_k = k - displs[owner]; // Индекс строки внутри процесса-владельца

        // Владелец k-й строки копирует её в буфер pivot_row
        if (rank == owner) {
            for (int j = 0; j < N; ++j) pivot_row[j] = local_A[local_k * N + j];
            pivot_row[N] = local_b[local_k];
        }

        // Рассылка ведущей строки всем процессам для выполнения исключения элементов
        MPI_Bcast(pivot_row.data(), N+1, MPI_DOUBLE, owner, MPI_COMM_WORLD);

        // Каждый процесс модифицирует только те свои строки, которые находятся ниже k-й
        for (int i = 0; i < local_n; ++i) {
            int global_row = displs[rank] + i; // Индекс строки в глобальной системе координат
            if (global_row <= k) continue;     // Пропускаем строки выше или равные текущей k

            // Вычисление множителя для обнуления k-го элемента текущей строки
            double factor = local_A[i*N + k] / pivot_row[k];
            // Вычитание ведущей строки, умноженной на factor, из локальной строки
            for (int j = k; j < N; ++j)
                local_A[i*N + j] -= factor * pivot_row[j];
            // Аналогичное преобразование для элемента локального вектора b
            local_b[i] -= factor * pivot_row[N];
        }
    }

    // Сборка обработанных частей матрицы A обратно на процесс 0 (теперь она треугольная)
    MPI_Gatherv(local_A.data(), local_n * N, MPI_DOUBLE,
                A_flat.data(), sendcounts_A.data(), displs_A.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Сборка измененных частей вектора b обратно на процесс 0
    MPI_Gatherv(local_b.data(), local_n, MPI_DOUBLE,
                b.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Вычисление финального решения (обратный ход) выполняется только процессом 0
    if (rank == 0) {
        vector<double> x(N); // Вектор неизвестных
        // Идем от последней строки к первой
        for (int i = N-1; i >= 0; --i) {
            x[i] = b[i]; // Начинаем со значения правой части
            // Вычитаем уже найденные значения неизвестных с их коэффициентами
            for (int j = i+1; j < N; ++j)
                x[i] -= A_flat[i*N + j] * x[j];
            // Делим на диагональный коэффициент
            x[i] /= A_flat[i*N + i];
        }

        // Вывод вектора решения x
        cout << "Solution vector x:" << endl;
        for (int i = 0; i < N; ++i) cout << fixed << setprecision(3) << x[i] << " ";
        cout << endl;

        // Фиксация времени окончания и вывод длительности работы программы
        double end_time = MPI_Wtime();
        cout << "Execution time: " << end_time - start_time << " seconds." << endl;
    }

    // Завершение работы с библиотекой MPI
    MPI_Finalize();
    return 0;
}
