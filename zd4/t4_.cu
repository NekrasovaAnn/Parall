#include <iostream>

#include <cub/cub.cuh>
#include <cub/block/block_reduce.cuh>

#include "cuda_runtime.h"

#include "sub.cuh" // contains functions for processing arguments and displaying them


#define at(arr, x, y) (arr[(x) * (n) + (y)])

// МАксимальное количество потоков для одного блока вычичлений
constexpr int MAXIMUM_THREADS_PER_BLOCK = 32;

// Cornerns
constexpr int LEFT_UP = 10;
constexpr int LEFT_DOWN = 20;
constexpr int RIGHT_UP = 20;
constexpr int RIGHT_DOWN = 30;

// Other values
constexpr int ITERS_BETWEEN_UPDATE = 400;

// Function definitions
void initArrays(double *mainArr, double *main_D, double *sub_D, cmdArgs *args);

__global__ void iterate(double *F, double *Fnew, double*, const cmdArgs *args);


int main(int argc, char *argv[]){
    cudaSetDevice(2); // selecting free GPU device
    // Аргуметы коммандной строки
    cmdArgs args = cmdArgs{false, false, 1E-6, (int)1E6, 10, 10}; // create default command line arguments
    processArgs(argc, argv, &args);
    printSettings(&args);

    double *F_H;  //Массив на процессоре
    double *F_D, *Fnew_D;  //Массивы на CUDA
    size_t size = args.n * args.m * sizeof(double);
    double error = 0;  //Ошибка
    int iterationsElapsed = 0;  //Реальное количество итераций
    
    //Выделяем памят на видеокарте
    cudaMalloc(&F_D, size);
    cudaMalloc(&Fnew_D, size);

    F_H = (double *)calloc(sizeof(double), size);

    //Заполняем массивы
    initArrays(F_H, F_D, Fnew_D, &args);

    {
        size_t grid_size = args.n * args.m;

        //Аргументы коммандной строки на видеокарту
        cmdArgs *args_d;
        cudaMalloc(&args_d, sizeof(cmdArgs));
        cudaMemcpy(args_d, &args, sizeof(cmdArgs), cudaMemcpyHostToDevice);

        //Массив для ошибок и ошибка на видеокарте
        double* substractions;
        cudaMalloc(&substractions, size);
        double* error_d;
        cudaMalloc(&error_d, sizeof(double));

        //Подготовка графа
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        cudaGraph_t graph;
        cudaGraphExec_t graph_instance;

        //Подготовка редукции CUB
        void *d_temp_storage = nullptr;  //Временная память для CUB
        size_t temp_storage_bytes = 0;
        cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, substractions, error_d, grid_size, stream);
        cudaMalloc(&d_temp_storage, temp_storage_bytes);


        //Начинаем строить граф
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

        //Количество потоков в одном блоке
        dim3 threadPerBlock {(unsigned int)(args.n + MAXIMUM_THREADS_PER_BLOCK - 1)/MAXIMUM_THREADS_PER_BLOCK,
                            (unsigned int)(args.m + MAXIMUM_THREADS_PER_BLOCK - 1)/MAXIMUM_THREADS_PER_BLOCK};
        if(threadPerBlock.x > MAXIMUM_THREADS_PER_BLOCK){
            threadPerBlock.x = MAXIMUM_THREADS_PER_BLOCK;
        }
        if(threadPerBlock.y > MAXIMUM_THREADS_PER_BLOCK){
            threadPerBlock.y = MAXIMUM_THREADS_PER_BLOCK;
        } 
        //Считаем количество блоков вычисления
        dim3 blocksPerGrid {(args.n + threadPerBlock.x - 1)/threadPerBlock.x,
                            (args.m + threadPerBlock.y - 1)/threadPerBlock.y};


        //Создаем граф
        for (size_t i = 0; i < ITERS_BETWEEN_UPDATE / 2; i++) {
            iterate<<<blocksPerGrid, threadPerBlock, 0, stream>>>(F_D, Fnew_D, substractions, args_d);
            iterate<<<blocksPerGrid, threadPerBlock, 0, stream>>>(Fnew_D, F_D, substractions, args_d);
        }

        cudaStreamEndCapture(stream, &graph);
        cudaGraphInstantiate(&graph_instance, graph, NULL, NULL, 0);

        //Основные вычисления
        do {
            //Проход по массиву
            cudaGraphLaunch(graph_instance, stream);
            cudaDeviceSynchronize();

            //Расчет ошибки
            cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, substractions, error_d, grid_size, stream);
            
            cudaMemcpy(&error, error_d, sizeof(double), cudaMemcpyDeviceToHost);  //Копируем ошибку на процессор
            iterationsElapsed += ITERS_BETWEEN_UPDATE; //Сколько итерация прошло
        } while (error > args.eps && iterationsElapsed < args.iterations);

        cudaGraphDestroy(graph);
        cudaStreamDestroy(stream);
        
    }


    std::cout << "Iterations: " << iterationsElapsed << std::endl;
    std::cout << "Error: " << error << std::endl;
    if (args.showResultArr) {
    cudaMemcpy(F_H, Fnew_D, size, cudaMemcpyDeviceToHost);
    int n = args.n;
        for (int x = 0; x < args.n; x++) {
            for (int y = 0; y < args.m; y++) {
                std::cout << at(F_H, x, y) << ' ';
            }
            std::cout << std::endl;
        }
    }

    cudaFree(F_D);
    cudaFree(Fnew_D);
    free(F_H);
    return 0;
}

//Заполнение массива
void initArrays(double *mainArr, double *main_D, double *sub_D, cmdArgs *args){
    int n = args->n;
    int m = args->m;
    size_t size = n * m * sizeof(double);

    //Углы
    at(mainArr, 0, 0) = LEFT_UP;
    at(mainArr, 0, m - 1) = RIGHT_UP;
    at(mainArr, n - 1, 0) = LEFT_DOWN;
    at(mainArr, n - 1, m - 1) = RIGHT_DOWN;

    //Края
    for (int i = 1; i < n - 1; i++) {
        at(mainArr, 0, i) = (at(mainArr, 0, m - 1) - at(mainArr, 0, 0)) / (m - 1) * i + at(mainArr, 0, 0);
        at(mainArr, i, 0) = (at(mainArr, n - 1, 0) - at(mainArr, 0, 0)) / (n - 1) * i + at(mainArr, 0, 0);
        at(mainArr, n - 1, i) = (at(mainArr, n - 1, m - 1) - at(mainArr, n - 1, 0)) / (m - 1) * i + at(mainArr, n - 1, 0);
        at(mainArr, i, m - 1) = (at(mainArr, n - 1, m - 1) - at(mainArr, 0, m - 1)) / (m - 1) * i + at(mainArr, 0, m - 1);
    }
    cudaMemcpy(main_D, mainArr, size, cudaMemcpyHostToDevice);
    cudaMemcpy(sub_D, mainArr, size, cudaMemcpyHostToDevice);
}

//Пересчет матрицы и ошибки
__global__ void iterate(double* F, double* Fnew, double* subs, const cmdArgs* args){

    int j = blockIdx.x * blockDim.x + threadIdx.x;    //Номер блока, размер блока, номер потока
	int i = blockIdx.y * blockDim.y + threadIdx.y;

    if(j == 0 || i == 0 || i == args->n-1 || j == args->n-1) return;  //Не трогаем стороны

    int n = args->n;
    at(Fnew, i, j) = 0.25 * (at(F, i+1, j) + at(F, i-1, j) + at(F, i, j+1) + at(F, i, j-1));  //Пересчитывание матрицы
    at(subs, i, j) = fabs(at(Fnew, i, j) - at(F, i, j));  //Расчет ошибки
}