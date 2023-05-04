#include <iostream>
#include <cstring>
#include <sstream>
#include <chrono>

#include <openacc.h>

#include </opt/nvidia/hpc_sdk/Linux_x86_64/22.11/cuda/11.8/targets/x86_64-linux/include/nvtx3/nvToolsExt.h>
#include <cublas_v2.h>

#define IDX2C(i, j, ld) (((j)*(ld))+(i))

// cublas API error checking
#define CUBLAS_CHECK(err)                                                                          \
    do {                                                                                           \
        cublasStatus_t err_ = (err);                                                               \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                                       \
            std::printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__);                   \
            throw std::runtime_error("cublas error");                                              \
        }                                                                                          \
    } while (0)

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

void print_array(double *A, int size)
{
    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            // Значение с GPU
            #pragma acc kernels present(A)
            printf("%.2f\t", A[IDX2C(i, j, size)]);
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}


int main(int argc, char *argv[]) {
    acc_set_device_num(2,acc_device_default);

    auto start = std::chrono::high_resolution_clock::now();
    int N = 128;
    int num_of_iter = 1000000;
    double accuracy = 0.000001;

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

    // Создаем указатель на структуру, содержащую контекст
    cublasHandle_t handle;
     // Инициализация контекста
    CUBLAS_CHECK(cublasCreate(&handle));
    
    double *arr = new double[N*N];
    double *arr2 = new double[N*N];
    double *arrD = new double[N*N];


    double delta = 10.0 / (N-1); // !
    int real_number_of_iteration = 0;

#pragma acc enter data create(delta, arr[0:N*N], arr2[0:N*N], arrD[0:N*N])
#pragma acc kernels
{
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
    memcpy(arr2, arr, N*N*sizeof(double));
}

    // Скаляр для вычитания
    const double alpha = -1;
    // Инкремент для матриц, в этой задаче 1
    const int inc = 1;
    // Индекс максимального элемента
    int max_idx = 0;

    delta = 0.0;
#pragma acc enter data copyin(arr[0:N*N], arr2[0:N*N], delta)
    for (int t = 0; t < num_of_iter; t++){
        real_number_of_iteration = t + 1;

#pragma acc kernels loop independent collapse(2) present(arr, arr2) async(1)
        for (int i = 1; i < N - 1; i++){
            for (int j = 1; j < N - 1; j++){
                arr2[IDX2C(i, j, N)] = (arr[IDX2C(i + 1, j, N)] + arr[IDX2C(i - 1, j, N)] + arr[IDX2C(i, j - 1, N)] + arr[IDX2C(i, j + 1, N)]) * 0.25;
            }
        }
        double *temp = arr;
        arr = arr2;
        arr2 = temp;

        //if (!(t % N)){
        #pragma acc kernels present(delta)
            delta = 0.0;
        #pragma acc data present(arr, arr2) deviceptr(arrD)
            {
                #pragma acc host_data use_device(arr, arr2, arrD)
                {
                    // arrD = arr2
                    CUBLAS_CHECK(cublasDcopy(handle, N*N, arr2, inc, arrD, inc));
                    // arrD = -1 * arr + arrD 
                    CUBLAS_CHECK(cublasDaxpy(handle, N*N, &alpha, arr, inc, arrD, inc));
                    // Получить индекс максимального абсолютного значения в arrD
                    CUBLAS_CHECK(cublasIdamax(handle, N*N, arrD, inc, &max_idx));
                    #pragma acc kernels present(delta)
                    delta = fabs(arrD[max_idx - 1]); // Fortran moment
                }
            }
       // }
                    //print_array(arr, N);
                    
//printf("CPU before %0.2f\n", delta);
#pragma acc update host (delta) wait(1)
        //printf("CPU after %0.2f\n", delta);
        if (delta < accuracy) break;
    }
     #pragma acc wait(1)

    cublasDestroy(handle);
#pragma acc exit data delete(arr[0:N*N],arr2[0:N*N], delta)
    std::cout << "Number of iteration: " << real_number_of_iteration << "\nAccuracy: " << delta << std::endl;

    delete[] arr;
    delete[] arr2;

    auto elapsed = std::chrono::high_resolution_clock::now() - start;
	long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	printf("%lld\n", microseconds/1000);

    return 0;
}