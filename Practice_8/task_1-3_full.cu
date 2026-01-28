%%writefile practice8_all.cu
#include <iostream>
#include <vector>
#include <random>
#include <omp.h>           // Библиотека для параллельных вычислений на CPU (OpenMP)
#include <cuda_runtime.h>  // Основной заголовочный файл CUDA

#define N 1000000          // Размер массива (1 миллион элементов)
#define RAND_MIN_VAL -100  // Минимальное значение для генерации
#define RAND_MAX_VAL 100   // Максимальное значение для генерации
#define BLOCK_SIZE 256     // Количество потоков в одном блоке CUDA

using namespace std;

// ================= CPU sequential (Последовательная обработка) =================
void processCPUSequential(float* data, int n) {
    for (int i = 0; i < n; i++) {
        data[i] *= 2.0f; // Умножаем каждый элемент на 2 в одном потоке
    }
}

// ================= CPU + OpenMP (Параллельная обработка на ядрах CPU) =================
void processCPUOpenMP(float* data, int n) {
    #pragma omp parallel for // Директива для распределения итераций цикла между потоками CPU
    for (int i = 0; i < n; i++) {
        data[i] *= 2.0f;
    }
}

// ================= CUDA kernel (Функция, выполняемая на видеокарте) =================
__global__ void processGPU(float* d_data, int n) {
    // Вычисляем глобальный индекс потока (Global ID)
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Проверка, чтобы не выйти за пределы массива
    if (gid < n) {
        d_data[gid] *= 2.0f;
    }
}

// ================= Print helper (Вспомогательная функция печати) =================
void printFirstElements(const vector<float>& data, const string& title) {
    cout << title << ": ";
    for (int i = 0; i < 5; i++) {
        cout << data[i] << " "; // Вывод первых 5 элементов для проверки корректности
    }
    cout << endl;
}

int main() {
    // ================= Data generation (Генерация исходных данных) =================
    vector<float> data(N);

    random_device rd;  // Источник энтропии
    mt19937 gen(rd()); // Генератор случайных чисел (Вихрь Мерсенна)
    uniform_int_distribution<> dist(RAND_MIN_VAL, RAND_MAX_VAL);

    for (int i = 0; i < N; i++) {
        data[i] = static_cast<float>(dist(gen)); // Заполнение массива
    }

    // Создание копий данных для каждого метода тестирования
    vector<float> data_seq = data;
    vector<float> data_omp = data;
    vector<float> data_gpu = data;
    vector<float> data_hybrid = data;

    printFirstElements(data, "Before processing");

    // ================= 1. Sequential CPU (Замер времени последовательного CPU) =================
    double t1 = omp_get_wtime(); // Начало отсчета (в секундах)
    processCPUSequential(data_seq.data(), N);
    double t2 = omp_get_wtime(); // Конец отсчета

    // ================= 2. CPU + OpenMP (Замер времени параллельного CPU) =================
    double t3 = omp_get_wtime();
    processCPUOpenMP(data_omp.data(), N);
    double t4 = omp_get_wtime();

    // ================= 3. GPU (CUDA) (Замер времени на видеокарте) =================
    float* d_data; // Указатель на память в видеокарте (Device memory)
    // Выделение памяти на GPU
    cudaMalloc(&d_data, N * sizeof(float));

    // Копирование данных из оперативной памяти (Host) в видеопамять (Device)
    cudaMemcpy(d_data, data_gpu.data(),
               N * sizeof(float),
               cudaMemcpyHostToDevice);

    int threads = BLOCK_SIZE; // Потоков в блоке
    int blocks = (N + threads - 1) / threads; // Расчет необходимого кол-ва блоков

    // Создание событий CUDA для точного замера времени выполнения ядра
    cudaEvent_t gstart, gstop;
    cudaEventCreate(&gstart);
    cudaEventCreate(&gstop);

    cudaEventRecord(gstart); // Старт записи времени GPU
    processGPU<<<blocks, threads>>>(d_data, N); // Запуск ядра на исполнение
    cudaEventRecord(gstop);  // Стоп записи времени GPU
    cudaEventSynchronize(gstop); // Ожидание завершения всех операций на GPU

    float gpuTime = 0.0f;
    cudaEventElapsedTime(&gpuTime, gstart, gstop); // Расчет времени в миллисекундах

    // Копирование результата обратно с GPU в оперативную память
    cudaMemcpy(data_gpu.data(), d_data,
               N * sizeof(float),
               cudaMemcpyDeviceToHost);

    // ================= 4. Hybrid CPU + GPU (Гибридный режим) =================
    int half = N / 2; // Делим задачу пополам

    double hstart = omp_get_wtime();

    // GPU часть: копируем только вторую половину данных
    cudaMemcpy(d_data,
               data_hybrid.data() + half,
               half * sizeof(float),
               cudaMemcpyHostToDevice);

    int hblocks = (half + threads - 1) / threads;
    // Запускаем GPU для обработки второй половины
    processGPU<<<hblocks, threads>>>(d_data, half);

    // В это же время CPU обрабатывает первую половину (OpenMP)
    processCPUOpenMP(data_hybrid.data(), half);

    // Ждем, пока видеокарта закончит свою часть работы
    cudaDeviceSynchronize();

    // Копируем результат GPU-части обратно в массив
    cudaMemcpy(data_hybrid.data() + half,
               d_data,
               half * sizeof(float),
               cudaMemcpyDeviceToHost);

    double hend = omp_get_wtime();

    // ================= Output (Вывод результатов) =================
    cout << endl;
    printFirstElements(data_seq,    "After Sequential CPU");
    printFirstElements(data_omp,    "After CPU + OpenMP");
    printFirstElements(data_gpu,    "After CUDA");
    printFirstElements(data_hybrid, "After Hybrid");

    cout << endl;
    cout << "Sequential CPU time:       " << (t2 - t1) * 1000 << " ms" << endl;
    cout << "CPU + OpenMP time:         " << (t4 - t3) * 1000 << " ms" << endl;
    cout << "CUDA kernel time:          " << gpuTime << " ms" << endl;
    cout << "Hybrid CPU + GPU time:     " << (hend - hstart) * 1000 << " ms" << endl;

    // ================= Cleanup (Освобождение ресурсов) =================
    cudaFree(d_data);         // Освобождение видеопамяти
    cudaEventDestroy(gstart); // Удаление объектов событий
    cudaEventDestroy(gstop);

    return 0;
}
