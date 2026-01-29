%%writefile practice_10_task_2.cu
#include <cuda_runtime.h> // Заголовочный файл среды исполнения CUDA
#include <omp.h>          // Интерфейс OpenMP (используется здесь для замера времени на CPU)
#include <iostream>       // Стандартный ввод-вывод
#include <iomanip>        // Манипуляторы форматирования вывода (setprecision)
#include <random>         // Генераторы случайных чисел C++11
#include <cmath>          // Математические функции

using namespace std;

// Константы для диапазона случайных чисел
#define RAND_MIN_VAL -100
#define RAND_MAX_VAL 100

// ---------------------------
// Печать первых 10 элементов массива для визуальной проверки
// ---------------------------
void print_edges(const float* data, int N) {
    cout << "First 10 elements: ";
    for (int i = 0; i < min(N, 10); i++) // Цикл до 10 или до конца массива, если он меньше
        cout << data[i] << " ";
    cout << endl;
}

// ---------------------------
// CUDA ядро (kernel): расчет частичной суммы квадратов отклонений
// ---------------------------
__global__ void kernel_variance(
        const float* data,     // Входной массив данных на GPU
        float mean,            // Среднее значение (вычислено на CPU)
        float* partial_sum,    // Указатель на итоговую сумму квадратов на GPU
        int N) {               // Размер массива

    // Вычисление глобального индекса потока
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // Шаг сетки (общее количество потоков), используется для обработки массивов > количества потоков
    int stride = blockDim.x * gridDim.x;

    float local = 0.0f; // Локальная переменная потока для накопления суммы
    // Каждый поток суммирует элементы с шагом stride (grid-stride loop)
    for (int i = tid; i < N; i += stride) {
        float diff = data[i] - mean; // Отклонение от среднего
        local += diff * diff;        // Квадрат отклонения
    }

    // Атомарное сложение: безопасно добавляет результат потока в общую переменную в глобальной памяти
    atomicAdd(partial_sum, local);
}

// ---------------------------
// Основная функция
// ---------------------------
int main() {

    // Вектор размеров массивов для тестирования масштабируемости
    vector<int> sizes = {10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000};
    int threads_per_block = 256; // Количество потоков в одном блоке CUDA

    // Инициализация генератора случайных чисел
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> dist(RAND_MIN_VAL, RAND_MAX_VAL);

    // Установка формата вывода чисел с плавающей точкой
    cout << fixed << setprecision(6);

    // Цикл по всем заданным размерам массивов
    for (int N : sizes) {

        cout << "\nArray size: " << N << endl;

        // ---------------------------
        // Pinned (закрепленная) хост-память
        // ---------------------------
        float* h_data;
        // Выделение памяти на CPU, которая не может быть вытеснена в файл подкачки (ускоряет копирование H2D)
        cudaMallocHost(&h_data, N * sizeof(float));

        // Заполнение массива случайными числами
        for (int i = 0; i < N; i++)
            h_data[i] = static_cast<float>(dist(gen));

        print_edges(h_data, N);

        // ---------------------------
        // CPU: расчет суммы и среднего значения
        // ---------------------------
        double cpu_start = omp_get_wtime(); // Фиксация времени начала на CPU

        double sum = 0.0;
        for (int i = 0; i < N; i++)
            sum += h_data[i];       // Нахождение суммы элементов
        double mean = sum / N;      // Среднее арифметическое

        double cpu_end = omp_get_wtime();   // Фиксация времени окончания
        double cpu_time = cpu_end - cpu_start;

        // ---------------------------
        // Память на GPU и поток (stream)
        // ---------------------------
        float *d_data, *d_var;
        cudaMalloc(&d_data, N * sizeof(float)); // Выделение памяти под массив на видеокарте
        cudaMalloc(&d_var, sizeof(float));      // Выделение памяти под одну переменную результата
        cudaMemset(d_var, 0, sizeof(float));    // Обнуление результата перед вычислениями

        cudaStream_t stream;
        cudaStreamCreate(&stream); // Создание неблокирующего потока CUDA для операций

        // ---------------------------
        // События CUDA для замера времени
        // ---------------------------
        cudaEvent_t h2d_start, h2d_stop;     // Копирование Host -> Device
        cudaEvent_t kernel_start, kernel_stop; // Работа ядра
        cudaEvent_t d2h_start, d2h_stop;     // Копирование Device -> Host

        // Инициализация всех событий
        cudaEventCreate(&h2d_start);
        cudaEventCreate(&h2d_stop);
        cudaEventCreate(&kernel_start);
        cudaEventCreate(&kernel_stop);
        cudaEventCreate(&d2h_start);
        cudaEventCreate(&d2h_stop);

        // ---------------------------
        // H2D копирование (Host to Device)
        // ---------------------------
        cudaEventRecord(h2d_start, stream); // Старт замера времени копирования
        // Асинхронное копирование данных в видеопамять
        cudaMemcpyAsync(d_data, h_data,
                        N * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
        cudaEventRecord(h2d_stop, stream);  // Конец замера

        // ---------------------------
        // Запуск ядра (Kernel)
        // ---------------------------
        // Расчет необходимого количества блоков
        int blocks = (N + threads_per_block - 1) / threads_per_block;

        cudaEventRecord(kernel_start, stream); // Старт замера времени работы ядра
        // Запуск функции на GPU в указанном потоке
        kernel_variance<<<blocks, threads_per_block, 0, stream>>>(
                d_data, (float)mean, d_var, N);
        cudaEventRecord(kernel_stop, stream);  // Конец замера

        // ---------------------------
        // D2H копирование (Device to Host)
        // ---------------------------
        float gpu_var_sum = 0.0f;

        cudaEventRecord(d2h_start, stream); // Старт замера времени обратного копирования
        // Копирование результата (суммы квадратов) обратно на CPU
        cudaMemcpyAsync(&gpu_var_sum, d_var,
                        sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
        cudaEventRecord(d2h_stop, stream);  // Конец замера

        // Ожидание завершения всех операций в потоке перед выводом данных
        cudaStreamSynchronize(stream);

        // ---------------------------
        // Расчет временных интервалов
        // ---------------------------
        float t_h2d, t_kernel, t_d2h;
        // Получение разницы во времени между событиями (в миллисекундах)
        cudaEventElapsedTime(&t_h2d, h2d_start, h2d_stop);
        cudaEventElapsedTime(&t_kernel, kernel_start, kernel_stop);
        cudaEventElapsedTime(&t_d2h, d2h_start, d2h_stop);

        // Вычисление финальной дисперсии
        double variance = gpu_var_sum / N;

        // Перевод времени GPU из мс в секунды
        double gpu_time =
                (t_h2d + t_kernel + t_d2h) / 1000.0;
        // Общее время: CPU (сумма/среднее) + GPU (копирование/ядро/копирование)
        double total_time = cpu_time + gpu_time;

        // ---------------------------
        // Вывод результатов
        // ---------------------------
        cout << "Sum:      " << sum << endl;
        cout << "Mean:     " << mean << endl;
        cout << "Variance: " << variance << endl;

        cout << "CPU time (s):        " << cpu_time << endl;
        cout << "H2D time (s):        " << t_h2d / 1000.0 << endl;
        cout << "Kernel time (s):     " << t_kernel / 1000.0 << endl;
        cout << "D2H time (s):        " << t_d2h / 1000.0 << endl;
        cout << "GPU total time (s):  " << gpu_time << endl;
        cout << "Hybrid total time (s): " << total_time << endl;
        cout << "----------------------------------------\n";

        // ---------------------------
        // Очистка ресурсов
        // ---------------------------
        cudaFree(d_data);       // Освобождение памяти на GPU
        cudaFree(d_var);
        cudaFreeHost(h_data);   // Освобождение pinned памяти на CPU

        cudaStreamDestroy(stream); // Уничтожение потока

        // Уничтожение событий
        cudaEventDestroy(h2d_start);
        cudaEventDestroy(h2d_stop);
        cudaEventDestroy(kernel_start);
        cudaEventDestroy(kernel_stop);
        cudaEventDestroy(d2h_start);
        cudaEventDestroy(d2h_stop);
    }

    return 0; // Завершение программы
}
