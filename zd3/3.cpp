#include <iostream>
#include <cstring>
#include <sstream>
#include <chrono>

#include <openacc.h>

#define IDX2C(i, j, ld) (((j)*(ld))+(i))

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


int main(int argc, char *argv[]) {
    acc_set_device_num(2,acc_device_default);

    auto start = std::chrono::high_resolution_clock::now();
    int N = 124;
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

    double *arr = new double[N*N];
    double *arr2 = new double[N*N];


    double delta = 10.0 / (N-1); // !
    int real_number_of_iteration = 0;

#pragma acc enter data create(delta, arr[0:N*N], arr2[0:N*N])
#pragma acc kernels
{
    arr[IDX2C(0, 0, N)] = 10.0;
    arr[IDX2C(0, N - 1, N)] = 20.0;
    arr[IDX2C(N - 1, 0, N)] = 20.0;
    arr[IDX2C(N - 1, N - 1, N)] = 30.0;

    arr2[IDX2C(0, 0, N)] = 10.0;
    arr2[IDX2C(0, N - 1, N)] = 20.0;
    arr2[IDX2C(N - 1, 0, N)] = 20.0;
    arr2[IDX2C(N - 1, N - 1, N)] = 30.0;


    for(int i = 1; i < N - 1; i++){
        arr[IDX2C(0, i, N)] = arr[IDX2C(0, i-1, N)] + delta;
        arr[IDX2C(N - 1, i, N)] = arr[IDX2C(N - 1, i-1, N)] + delta;
        arr[IDX2C(i, 0, N)] = arr[IDX2C(i-1, 0, N)] + delta;
        arr[IDX2C(i, N - 1, N)] = arr[IDX2C(i-1, N - 1, N)] + delta;
        arr2[IDX2C(0, i, N)] = arr2[IDX2C(0, i-1, N)] + delta;
        arr2[IDX2C(N - 1, i, N)] = arr2[IDX2C(N - 1, i-1, N)] + delta;
        arr2[IDX2C(i, 0, N)] = arr2[IDX2C(i-1, 0, N)] + delta;
        arr2[IDX2C(i, N - 1, N)] = arr2[IDX2C(i-1, N - 1, N)] + delta;
    }
}


    for (int t = 0; t < num_of_iter; t++){
        real_number_of_iteration = t + 1;
        #pragma acc kernels present(delta) // !
        delta = 0.0;
//#pragma acc update device (delta)        // !
#pragma acc parallel loop collapse(2) reduction(max:delta) present(arr, arr2)
        for (int i = 1; i < N - 1; i++){
            for (int j = 1; j < N - 1; j++){
                arr2[IDX2C(i, j, N)] = (arr[IDX2C(i + 1, j, N)] + arr[IDX2C(i - 1, j, N)] + arr[IDX2C(i, j - 1, N)] + arr[IDX2C(i, j + 1, N)]) * 0.25;
                delta = std::max(delta, std::abs(arr2[IDX2C(i, j, N)] - arr[IDX2C(i, j, N)]));
            }
        }

        double *temp = arr;
        arr = arr2;
        arr2 = temp;
        //printf("%lf\n", delta);
#pragma acc update host (delta)
        //printf("%lf\n", delta);
        if (delta < accuracy) break;
    }

#pragma acc exit data delete(arr[0:N*N],arr2[0:N*N]) copyout(delta)
    std::cout << "Number of iteration: " << real_number_of_iteration << "\nAccuracy: " << delta << std::endl;

    delete[] arr;
    delete[] arr2;

    auto elapsed = std::chrono::high_resolution_clock::now() - start;
	long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	printf("%lld\n", microseconds/1000);

    return 0;
}