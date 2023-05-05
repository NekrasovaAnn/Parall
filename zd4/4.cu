#include <stdio.h>
#include <time.h>
#include <sstream>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#define IDX2C(i, j, ld) (((j)*(ld))+(i))

//расчет уравнения теплопроводности по блокам и потокам. Используем j-ый поток и i-ый блок при каждом расчете, принцип с четвертьюсумммой остается тем же ХУЙНЯ ЭТО ВСЕ
__global__ void heat_equation(double* arr, double* arr2, double dt, int N) {
    size_t i = blockIdx.x;
	size_t j = threadIdx.x;
    
	if(!(blockIdx.x == 0 || threadIdx.x == 0))
	{
        arr2[IDX2C(i, j, N)] = (arr[IDX2C(i + 1, j, N)] + arr[IDX2C(i - 1, j, N)] + arr[IDX2C(i, j - 1, N)] + arr[IDX2C(i, j + 1, N)]) * dt;
    }
}

//расчет ошибки по потокам, проставление его в итоговую матрицу
__global__ void get_error(double* arr, double* arr2, double* out)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx>0)
	{
		out[idx] = fabs(arr2[idx] - arr[idx]);
	}
}


void print_array(double *A, int size)
{
    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            printf("%.6f\t", A[IDX2C(i, j, size)]);
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}


template<typename T>
T extractNumber(char* arr){
    std::stringstream stream;
    stream << arr;
    T result;
    if (!(stream >> result)){
        throw std::invalid_argument("Wrong argument type");
    }
    return result;
}


int main(int argc, char** argv) {

    int N = 128;
    int num_of_iter = 1000000;
    double accuracy = 0.000001;

//Получаем параметры из командной строки
    for(int arg = 0; arg < argc; arg++){
        if(std::strcmp(argv[arg], "-eps") == 0){
            accuracy = extractNumber<double>(argv[arg+1]);
            arg++;
        }
        else if(std::strcmp(argv[arg], "-i") == 0){
            num_of_iter = extractNumber<int>(argv[arg+1]);
            arg++;
        }
        else if(std::strcmp(argv[arg], "-s") == 0){
            N = extractNumber<int>(argv[arg+1]);
            arg++;
        }
    }

//Начинаем отсчет времени
    clock_t start = clock();
    
//Объявляем массивы
    int size = N*N;
    double *arr = (double *)calloc(sizeof(double), size);
    double *arr2 = (double *)calloc(sizeof(double), size);

    double delta = 10.0 / (N-1);


//Заполняем массив
    arr[IDX2C(0, 0, N)] = 10.0;
    arr[IDX2C(0, N - 1, N)] = 20.0;
    arr[IDX2C(N - 1, 0, N)] = 20.0;
    arr[IDX2C(N - 1, N - 1, N)] = 30.0;

    for(int i = 1; i < N - 1; i++){
        arr[IDX2C(0, i, N)] = arr[IDX2C(0, i-1, N)] + delta;
        arr[IDX2C(N - 1, i, N)] = arr[IDX2C(N - 1, i-1, N)] + delta;
        arr[IDX2C(i, 0, N)] = arr[IDX2C(i-1, 0, N)] + delta;
        arr[IDX2C(i, N - 1, N)] = arr[IDX2C(i-1, N - 1, N)] + delta;
    }
    memcpy(arr2, arr, size);


//копируем из рассчитанной матрицы в новую

    memcpy(arr2, arr, sizeof(double) * size);
	

//объявление указателей под матрицы на устройстве, матрицы для ошибок  и буфер, а также последующее выделение памяти для них на устройстве

	
    double* Matrix, *MatrixNew, *Error, *deviceError, *errortemp = 0;

    cudaMalloc((&Matrix), size*sizeof(double));
    cudaMalloc((&MatrixNew), size*sizeof(double));
    cudaMalloc((&Error), size*sizeof(double));
    cudaMalloc((&deviceError), sizeof(double));


//копируем из рассчитанных матриц хоста на устройство

    cudaMemcpy(Matrix, arr, size*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(MatrixNew, arr2, size*sizeof(double), cudaMemcpyHostToDevice);

    size_t tempsize = 0;
    double dt = 0.25;
    int k = 0;
    double error = 1.0; 



//создание графа
 
	bool isGraphCreated = false;
	cudaStream_t stream, memoryStream;
	cudaStreamCreate(&stream);
	cudaStreamCreate(&memoryStream);
	cudaGraph_t graph;
	cudaGraphExec_t instance;
	

//функция редукции определяет, какой размер буфера ей понадобится и записывает его в tempsize
//следующей же строкой - выделение памяти
	
    cub::DeviceReduce::Max(errortemp, tempsize, Error, deviceError, size, stream);
    cudaMalloc((&errortemp), tempsize);

//максимальный размер блока - 1024, расчет размера блока в зависимости от GPU, в blockSize будет лежать оптимальный размер сетки

	 int blockS, minGridSize;
	 int maxSize = 1024; 
	 cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockS, heat_equation, 0, maxSize);
	 dim3 blockSize(blockS, 1);
	 dim3 gridSize((N-1)/blockSize.x + 1, (N-1)/blockSize.y + 1);
	
    for (; (k < num_of_iter) && (error > accuracy); ) { 
        printf("%d\n", k);
	    if (isGraphCreated) {
			cudaGraphLaunch(instance, stream);
			cudaStreamSynchronize(stream);
            cudaDeviceSynchronize();
			get_error<<<gridSize, blockSize, 0, stream>>>(Matrix, MatrixNew, Error);
			cub::DeviceReduce::Max(errortemp, tempsize, Error, deviceError, size, stream);
			cudaMemcpy(&error, deviceError, sizeof(double), cudaMemcpyDeviceToHost);

			k += 100;

            printf("+\n");
		}
			
		//вызов расчета функции теплопроводности на устройстве с размером сетки N-1 и таким же размером блока
	    else {
			cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
			for(size_t i = 0; i < 50; i++)
			{
				heat_equation<<<gridSize, blockSize, 0, stream>>>(Matrix, MatrixNew, dt, N);
				heat_equation<<<gridSize, blockSize, 0, stream>>>(MatrixNew, Matrix, dt, N);
			}
			
			//вызов нахождения ошибки на устройстве с тем же размером сетки и блока
	
			cudaStreamEndCapture(stream, &graph);
			cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
			isGraphCreated = true;

            printf("f\n");
  		}
    }

//конец отсчета, вывод полученных данных

    clock_t end = clock();
    printf("Time is = %lf\n", 1.0*(end-start)/CLOCKS_PER_SEC);
    
    printf("%d %lf\n", k, error);

//очистка памяти

    cudaFree(Matrix);
    cudaFree(MatrixNew);
    cudaFree(Error);
    cudaFree(errortemp);
    free(arr);
    free(arr2);
    return 0;

}