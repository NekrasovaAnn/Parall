#include <mpi.h>

#include "cuda_runtime.h"

#include "sub.cuh"
#include "CudaFunc.cuh"


#ifdef DEBUG
#define DEBUG_PRINTF(line, a...) printf(line, ## a)
#else
#define DEBUG_PRINTF(line, a...) 0
#endif

#ifdef DEBUG1
#define DEBUG1_PRINTF(line, a...) printf(line, ## a)
#else
#define DEBUG1_PRINTF(line, a...) 0
#endif



//   Режимы распределения потоков по блокам
#ifdef THREAD_ANALOG_MODE
// Максимальное количество потоков для одного блока вычичлений
constexpr int MAXIMUM_THREADS_PER_BLOCK = 1024;
#define THREAD_PER_BLOCK_DEFINED(arr_lenght_dim_1, arr_lenght_dim_2, max_thread) (max_thread)
#define BLOCK_COUNT_DEFINED(arr_lenght_dim_1, arr_lenght_dim_2, threads_count) (arr_lenght_dim_2/threads_count.x?arr_lenght_dim_2/threads_count.x:1), arr_lenght_dim_1
#else
// Максимальное количество потоков для одного блока вычичлений
constexpr int MAXIMUM_THREADS_PER_BLOCK = 32;
#define THREAD_PER_BLOCK_DEFINED(arr_lenght_dim_1, arr_lenght_dim_2, max_thread) ((arr_lenght_dim_1+max_thread-1)/max_thread), ((arr_lenght_dim_2+max_thread-1)/max_thread)
#define BLOCK_COUNT_DEFINED(arr_lenght_dim_1, arr_lenght_dim_2, threads_count) ((arr_lenght_dim_1 + threads_count.x - 1) / threads_count.x), ((arr_lenght_dim_2 + threads_count.y - 1) / threads_count.y)
#endif


//   Углы
constexpr int LEFT_UP = 10;
constexpr int LEFT_DOWN = 20;
constexpr int RIGHT_UP = 20;
constexpr int RIGHT_DOWN = 30;

constexpr int THREADS_PER_BLOCK_REDUCE = 256;

constexpr int ITERS_BETWEEN_UPDATE = 400;


int main(int argc, char *argv[]) {
    //   Инициализируем MPI  
    int rank, ranks_count;
    MPI_Init(&argc, &argv);  //  Инициализирует среду выполнения вызывающего процесса MPI

    //   Определяем сколько процессов внутри глоабльного коммуникатора  
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  //  Извлекает ранг вызывающего процесса
    MPI_Comm_size(MPI_COMM_WORLD, &ranks_count);  //  Извлекает количество процессов, задействованных в коммуникаторе

    //   Каждый процесс выбирает свою видеокарту  
    cudaSetDevice(rank);

    //   Аргументы командной строки  
    cmdArgs global_args = cmdArgs{false, false, 1E-6, (int)1E6, 16, 16};
    //  Считываем
    processArgs(argc, argv, &global_args);

    if(rank == 0){
        //  Пишем
        printSettings(&global_args);
    }

    //   Расчет элементов для каждого процесса
    int TOTAL_GRID_SIZE = global_args.m * global_args.n;   //  Общий размер сетки

    cmdArgs local_args{global_args};
    local_args.n =  TOTAL_GRID_SIZE / ranks_count / global_args.m + 2 * (rank != ranks_count - 1);  // Количество строк в каждом процессе

    int ELEMENTS_BY_PROCESS = local_args.n * local_args.m;//  Количество элементов в каждом процессе

    //   Создание указателей на массивы  
    double *F_H_full = nullptr; // Указатель для хранения всего массива (используется в rank = 0)
    double *error_array = nullptr; // Указатель для хранения массива ошибок полученных с остальных процессов на нулевой (используется в rank 0)
    double *F_H;
    double *F_D, *Fnew_D;
    size_t array_size_bytes = ELEMENTS_BY_PROCESS * sizeof(double);

    //   Выделяем память для GPU  
    cudaMalloc(&F_D, array_size_bytes);
    cudaMalloc(&Fnew_D, array_size_bytes);
    //  CPU
    cudaMallocHost(&F_H, array_size_bytes);

    if(rank == 0){
        error_array = (double*)malloc(sizeof(double) * ranks_count);
    }

    
   
    int n = global_args.n;
    int m = global_args.m;

    // Заполняем полный массив в 0 процессе 
    if(rank == 0){
        // Иницилизируем массив в 0 процессе
        F_H_full = (double*)calloc(n*m, sizeof(double));

        for (int i = 0; i < n * m && global_args.initUsingMean; i++){
            F_H_full[i] = (LEFT_UP + LEFT_DOWN + RIGHT_UP + RIGHT_DOWN) / 4;
        }

        at(F_H_full, 0, 0) = LEFT_UP;
        at(F_H_full, 0, m - 1) = RIGHT_UP;
        at(F_H_full, n - 1, 0) = LEFT_DOWN;
        at(F_H_full, n - 1, m - 1) = RIGHT_DOWN;

        for (int i = 1; i < n - 1; i++) {
            at(F_H_full, 0, i) = (at(F_H_full, 0, m - 1) - at(F_H_full, 0, 0)) / (m - 1) * i + at(F_H_full, 0, 0);
            at(F_H_full, i, 0) = (at(F_H_full, n - 1, 0) - at(F_H_full, 0, 0)) / (n - 1) * i + at(F_H_full, 0, 0);
            at(F_H_full, n - 1, i) = (at(F_H_full, n - 1, m - 1) - at(F_H_full, n - 1, 0)) / (m - 1) * i + at(F_H_full, n - 1, 0);
            at(F_H_full, i, m - 1) = (at(F_H_full, n - 1, m - 1) - at(F_H_full, 0, m - 1)) / (m - 1) * i + at(F_H_full, 0, m - 1);
        }

        int data_start = 0;
        int data_lenght = 0;

        // Отправляем необходимые части всем процессам, включая самого себя
        // Каждый процесс обрабатывает local_args.n - строк
        // Это значение зависит от того какой участок обрабатывает наш процесс
        // В итоге получаем что каждый процесс обрабатывает global_args.n / 4 + 2 строк
        // Кроме последнего, он обрабатывает только global_args.n / 4 строк
        // Это происходит из-за того, что ему нет необходимости поддерживать граничные значения с нижним блоком (он является самым нижним)
        for(size_t target = 0; target < ranks_count; target++){
            MPI_Request req;
            data_lenght = ELEMENTS_BY_PROCESS - 2 * local_args.m * (target == (ranks_count - 1) && ranks_count != 1);
            DEBUG_PRINTF("Sended to %d elems: %d from: %d\n", target, data_lenght, data_start);
            MPI_Isend(
                F_H_full + data_start,
                data_lenght,
                MPI_DOUBLE,
                target,
                0,
                MPI_COMM_WORLD,
                &req
            );  //  Инициирует операцию отправки в стандартном режиме и возвращает дескриптор запрошенной операции связи.
            
            data_start += data_lenght - local_args.m * 2;
        }
        
        // Ждём получения обрабатываемой части от 0 процесса
        MPI_Status status;  //  Хранит сведения о полученом сообщении
        MPI_Recv(
            F_H,
            ELEMENTS_BY_PROCESS,
            MPI_DOUBLE,
            0,
            0,
            MPI_COMM_WORLD,
            &status);  //  Выполняет операцию получения и не возвращается до получения соответствующего сообщения

        cudaMemcpy(F_D, F_H, array_size_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(Fnew_D, F_H, array_size_bytes, cudaMemcpyHostToDevice);
    }

    double error = 1;
    int iterationsElapsed = 0;



    //  Аргументы коммандной строки на видеокарту
    cmdArgs *args_d;
    cudaMalloc(&args_d, sizeof(cmdArgs));
    cudaMemcpy(args_d, &local_args, sizeof(cmdArgs), cudaMemcpyHostToDevice);

    int num_blocks_reduce = (ELEMENTS_BY_PROCESS + THREADS_PER_BLOCK_REDUCE - 1) / THREADS_PER_BLOCK_REDUCE;

    //  Массив для ошибок и ошибка на видеокарте
    double *error_reduction;
    cudaMalloc(&error_reduction, sizeof(double) * num_blocks_reduce);
    double *error_d;
    cudaMalloc(&error_d, sizeof(double));

    //  Подготовка редукции CUB
    void *d_temp_storage = nullptr;  //  Временная память для CUB
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, error_reduction, error_d, num_blocks_reduce);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    //  Количество потоков в одном блоке
    dim3 threadPerBlock {THREAD_PER_BLOCK_DEFINED(local_args.n, local_args.m, MAXIMUM_THREADS_PER_BLOCK)}; 
    if(threadPerBlock.x > MAXIMUM_THREADS_PER_BLOCK){
        threadPerBlock.x = MAXIMUM_THREADS_PER_BLOCK;
    }
    if(threadPerBlock.y > MAXIMUM_THREADS_PER_BLOCK){
        threadPerBlock.y = MAXIMUM_THREADS_PER_BLOCK;
    } 
    //  Количество блоков вычисления
    dim3 blocksPerGrid {BLOCK_COUNT_DEFINED(local_args.n, local_args.m, threadPerBlock)};

    DEBUG_PRINTF("%d: %d %d %d %d\n", rank, threadPerBlock.x, threadPerBlock.y, blocksPerGrid.x, blocksPerGrid.y);

    MPI_Barrier(MPI_COMM_WORLD);  //  Инициирует синхронизацию барьеров для всех членов группы.
    
    do {

        iterate<<<blocksPerGrid, threadPerBlock, 0, stream>>>(F_D, Fnew_D, args_d);  //   Проход по своему участку

        //  Обмен граничными условиями
        transfer_data(rank, ranks_count, F_H, F_D, local_args, stream);

        iterate<<<blocksPerGrid, threadPerBlock, 0, stream>>>(Fnew_D, F_D, args_d);  //   Проход по своему участку

        // Обмен граничными условиями
        transfer_data(rank, ranks_count, F_H, Fnew_D, local_args, stream);
        
        iterationsElapsed += 2;
        //  Расчет ошибки
        if(iterationsElapsed % ITERS_BETWEEN_UPDATE == 0){ 
            block_reduce<<<num_blocks_reduce, THREADS_PER_BLOCK_REDUCE, 0, stream>>>(F_D, Fnew_D, ELEMENTS_BY_PROCESS, error_reduction);
            cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, error_reduction, error_d, num_blocks_reduce, stream);
            cudaStreamSynchronize(stream);  // Ожидание завершения потоковых задач
            cudaMemcpy(&error, error_d, sizeof(double), cudaMemcpyDeviceToHost);

            // Сборка ошибок с каждого процесса и обработка их на 0 потоке (Procces reduction)
            {
                MPI_Gather(&error, 1, MPI_DOUBLE, error_array, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);  //  Собирает данные от всех участников группы к одному участнику.

                if(rank == 0){
                    error = 0;
                    for(int err_id = 0; err_id < ranks_count; err_id++){
                        error = max(error, error_array[err_id]);  //  Вычисление максимальной ошибки
                    }
                    DEBUG1_PRINTF("iters: %d error: %lf\n", iterationsElapsed, error);
                }
                MPI_Barrier(MPI_COMM_WORLD);  //  Инициирует синхронизацию барьеров для всех членов группы.
                MPI_Bcast(&error, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);  //  Передает данные от одного участника группы всем членам группы
                                                                      //  маркер того, что основной процесс обработал все их отправленные данные
            }
        }
    } while(error > global_args.eps && iterationsElapsed < global_args.iterations);

    cudaStreamDestroy(stream);

    if(global_args.showResultArr){
        //  Отправка финального массива на нулевой процесс
        cudaMemcpy(F_H, F_D, array_size_bytes, cudaMemcpyDeviceToHost);
        MPI_Request req;
        MPI_Isend(F_H + local_args.m, ELEMENTS_BY_PROCESS - (local_args.m * 2), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &req);  //  Инициирует операцию отправки в стандартном режиме и возвращает дескриптор запрошенной операции связи.

        //  Собираем все части массива на нулевом процессе
        if(rank == 0){
            int array_offset = local_args.m;  //  Смещение массива
            for(int target = 0; target < ranks_count; target++){
                MPI_Status status;
                int recive_size = ELEMENTS_BY_PROCESS - 2 * local_args.m - 2 * local_args.m * (target == (ranks_count - 1));
                MPI_Recv(F_H_full + array_offset, recive_size, MPI_DOUBLE, target, 0, MPI_COMM_WORLD, &status);  //  Выполняет операцию получения 
                array_offset += recive_size;
            }

            //  Выводим массив
            std::cout << rank << " ---\n";
            for (int x = 0; x < global_args.n; x++) {
                    int n = global_args.m;
                    for (int y = 0; y < global_args.m; y++) {
                        std::cout << at(F_H_full, x, y) << ' ';
                    }
                    std::cout << std::endl;
                }
            std::cout << std::endl;
        }
    }

    if(rank == 0){
        std::cout << "Iterations: " << iterationsElapsed << std::endl;
        std::cout << "Error: " << error << std::endl;
    }

    //  Удаляем все, что выделили
    if(F_H_full) free(F_H_full);
    if(error_array) free(error_array);
    cudaFree(F_D);
    cudaFree(Fnew_D);
    cudaFree(F_H);

    MPI_Finalize();  //  Завершает среду выполнения вызывающего процесса MPI.
    return 0;
}