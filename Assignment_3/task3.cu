%%writefile assignment3_task3.cu
#include <cuda_runtime.h> // Библиотека среды выполнения CUDA
#include <iostream>       // Стандартный ввод-вывод
#include <random>         // Генерация случайных чисел

// Константы
#define N 1000000         // Количество элементов в массиве
#define BLOCK_SIZE 256    // Количество потоков в одном блоке
#define RAND_MIN_VAL 1    // Минимальное случайное значение
#define RAND_MAX_VAL 100  // Максимальное случайное значение

using namespace std;

// -----------------------------
// Макрос для автоматической проверки ошибок CUDA-функций
// -----------------------------
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        cerr << "CUDA error: " << cudaGetErrorString(err) \
             << " at " << __FILE__ << ":" << __LINE__ << endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// -----------------------------
// Ядра (Kernels)
// -----------------------------

// 1. Ядро с коалесцированным доступом (Эффективно)
// Потоки одного варпа (группа из 32 потоков) обращаются к соседним ячейкам памяти.
// Контроллер памяти может объединить эти запросы в одну транзакцию.
__global__ void coalesced_kernel(float* data, float factor, int n) {
    // Стандартное вычисление индекса: поток 0 -> элемент 0, поток 1 -> элемент 1
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) data[idx] *= factor;
}

// 2. Ядро с некоалесцированным доступом (Неэффективно)
// Здесь индекс вычисляется так, что соседние потоки читают данные из очень далеких участков памяти.
// Это заставляет видеокарту выполнять множество отдельных транзакций вместо одной.
__global__ void uncoalesced_kernel(float* data, float factor, int n) {
    // Нарочито "плохая" формула индексации
    int idx = threadIdx.x * gridDim.x + blockIdx.x; 
    if(idx < n) data[idx] *= factor;
}

// -----------------------------
// Вспомогательная функция для печати данных
// -----------------------------
void print_edges(const float* a, const string& msg) {
    cout << msg << "\nFirst 5: ";
    for(int i=0;i<5;i++) cout<<a[i]<<" "; // Печать начала
    cout<<"\nLast 5: ";
    for(int i=N-5;i<N;i++) cout<<a[i]<<" "; // Печать конца
    cout<<"\n\n";
}

// -----------------------------
// Основная функция
// -----------------------------
int main() {
    // Проверка наличия видеокарты
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if(deviceCount==0) { cerr<<"No CUDA GPU found\n"; return 1; }

    float *h_data, *h_res, *d_data; // Указатели для CPU (h_) и GPU (d_)
    size_t size = N*sizeof(float);  // Общий объем памяти в байтах
    float factor = 2.5f;            // Множитель

    // Инициализация случайных чисел на хосте
    random_device rd; 
    mt19937 gen(rd()); 
    uniform_int_distribution<> dist(RAND_MIN_VAL, RAND_MAX_VAL);
    
    h_data = (float*)malloc(size); // Выделение RAM
    h_res  = (float*)malloc(size); // Память для хранения результата после копирования
    for(int i=0;i<N;i++) h_data[i] = static_cast<float>(dist(gen));

    // Выделение видеопамяти
    CUDA_CHECK(cudaMalloc(&d_data, size));

    // Расчет количества блоков
    int grid = (N + BLOCK_SIZE -1)/BLOCK_SIZE;

    // Создание событий для замера времени
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float time_coalesced = 0.0f;
    float time_uncoalesced = 0.0f;

    // ========================
    // ТЕСТ 1: КОАЛЕСЦИРОВАННЫЙ ДОСТУП
    // ========================
    cout<<"=== COALESCED ACCESS ===\n";
    print_edges(h_data,"Before");

    // Копирование данных на GPU
    CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaEventRecord(start)); // Запуск секундомера
    coalesced_kernel<<<grid,BLOCK_SIZE>>>(d_data,factor,N); // Вызов ядра
    CUDA_CHECK(cudaEventRecord(stop));  // Остановка секундомера
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stop)); // Синхронизация CPU и GPU
    CUDA_CHECK(cudaEventElapsedTime(&time_coalesced,start,stop)); // Расчет времени в мс
    
    // Копирование результата обратно для проверки
    CUDA_CHECK(cudaMemcpy(h_res,d_data,size,cudaMemcpyDeviceToHost));
    print_edges(h_res,"After");
    cout<<"Time: "<<time_coalesced<<" ms\n\n";

    // ========================
    // ТЕСТ 2: НЕКОАЛЕСЦИРОВАННЫЙ ДОСТУП
    // ========================
    cout<<"=== UNCOALESCED ACCESS ===\n";
    print_edges(h_data,"Before");

    // Сброс данных на GPU в исходное состояние
    CUDA_CHECK(cudaMemcpy(d_data,h_data,size,cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaEventRecord(start));
    uncoalesced_kernel<<<grid,BLOCK_SIZE>>>(d_data,factor,N); // Вызов "плохого" ядра
    CUDA_CHECK(cudaEventRecord(stop));
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&time_uncoalesced,start,stop));
    
    CUDA_CHECK(cudaMemcpy(h_res,d_data,size,cudaMemcpyDeviceToHost));
    print_edges(h_res,"After");
    cout<<"Time: "<<time_uncoalesced<<" ms\n\n";

    // ========================
    // РАСЧЕТ ЗАМЕДЛЕНИЯ
    // ========================
    float slowdown = time_uncoalesced / time_coalesced;
    cout<<"===== PERFORMANCE COMPARISON =====\n";
    cout<<"Coalesced access time:   "<<time_coalesced<<" ms\n";
    cout<<"Uncoalesced access time: "<<time_uncoalesced<<" ms\n";
    cout<<"Slowdown (uncoalesced / coalesced): "<<slowdown<<"x\n\n";

    // Очистка памяти и уничтожение событий
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    cudaFree(d_data);
    free(h_data);
    free(h_res);

    return 0;
}
