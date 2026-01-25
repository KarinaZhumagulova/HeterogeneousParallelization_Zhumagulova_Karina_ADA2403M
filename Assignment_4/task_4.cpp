#include <mpi.h>        // Подключение библиотеки MPI для параллельного программирования
#include <iostream>     // Подключение библиотеки ввода-вывода
#include <vector>       // Подключение контейнера vector (динамический массив)
#include <chrono>       // Подключение библиотеки для замера времени
#include <random>       // Подключение генератора случайных чисел

int main(int argc, char** argv) {
    // Инициализация среды MPI: настройка коммуникаций и обработка аргументов командной строки
    MPI_Init(&argc, &argv);

    int rank, size;
    // Получение уникального номера текущего процесса (от 0 до size-1)
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // Получение общего количества запущенных процессов
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int N = 1000000;              // Общий размер глобального массива
    int chunk_size = N / size;          // Вычисление размера части массива для каждого процесса

    // Создание локального вектора, в котором каждый процесс будет хранить свою часть данных
    std::vector<int> local_array(chunk_size);

    // Инициализация генератора случайных чисел Мэрсенна-Твистера
    // Используем rank + 1 в качестве зерна (seed), чтобы у каждого процесса были свои числа
    std::mt19937 gen(rank + 1);
    // Настройка распределения: целые числа в диапазоне от 1 до 100
    std::uniform_int_distribution<int> dist(1, 100);

    // Заполнение локальной части массива случайными значениями
    for (int i = 0; i < chunk_size; i++)
        local_array[i] = dist(gen);

    // ---------------- Засекаем общее время на процессе 0 ----------------
    // Барьерная синхронизация: все процессы ждут здесь, пока последний не дойдет до этой точки
    MPI_Barrier(MPI_COMM_WORLD);
    // Фиксация времени начала вычислений
    auto start = std::chrono::high_resolution_clock::now();

    // ---------------- Локальная обработка: умножение на 2 ----------------
    // Каждый процесс независимо обрабатывает только свою часть массива в своей памяти
    for (int i = 0; i < chunk_size; i++)
        local_array[i] *= 2;

    // Вторая барьерная синхронизация: гарантируем, что все процессы закончили вычисления
    MPI_Barrier(MPI_COMM_WORLD);
    // Фиксация времени окончания вычислений
    auto end = std::chrono::high_resolution_clock::now();

    // ---------------- Сбор полного массива на процессе 0 ----------------
    std::vector<int> full_array;
    // Только процесс с рангом 0 выделяет память под весь массив N
    if(rank == 0) full_array.resize(N);


    // MPI_Gather собирает части local_array со всех процессов и склеивает их в full_array на процессе 0
    MPI_Gather(local_array.data(), chunk_size, MPI_INT,
               full_array.data(), chunk_size, MPI_INT,
               0, MPI_COMM_WORLD);

    // ---------------- Вывод результатов на процессе 0 ----------------
    if(rank == 0) {
        int print_count = 10; // Сколько элементов выводить для проверки

        // Демонстрация оригинального массива (путем обратного деления на 2)
        std::cout << "Original array - first " << print_count << ": ";
        for(int i = 0; i < print_count; i++)
            std::cout << full_array[i] / 2 << " ";
        std::cout << "\nOriginal array - last " << print_count << ": ";
        for(int i = N - print_count; i < N; i++)
            std::cout << full_array[i] / 2 << " ";
        std::cout << std::endl;

        // Вывод элементов массива после их параллельной обработки
        std::cout << "Processed array - first " << print_count << ": ";
        for(int i = 0; i < print_count; i++)
            std::cout << full_array[i] << " ";
        std::cout << "\nProcessed array - last " << print_count << ": ";
        for(int i = N - print_count; i < N; i++)
            std::cout << full_array[i] << " ";
        std::cout << std::endl;

        // Расчет и вывод затраченного времени в миллисекундах
        double total_time = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "Total processing time (ms): " << total_time << std::endl;
    }

    // Завершение работы MPI: очистка ресурсов и закрытие соединений между процессами
    MPI_Finalize();
    return 0;
}
