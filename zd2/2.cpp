#include <iostream>
#include <cstring>
#include <chrono>
#include <sstream>

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

double accuracy = 0.000001;
int num_of_iter = 1000000;

int main(int argc, char *argv[]) {
    auto start = std::chrono::high_resolution_clock::now();
    int N = 0;

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

    //std::cout<<"accuracy: "<<accuracy<<"\nnum_of_iter: "<<num_of_iter<<"\nN: "<<N<<std::endl;

    double** arr = new double*[N];
    double** arr2 = new double*[N];
    for (int i= 0;i<N;i++){
        arr[i] = new double[N];
        arr2[i] = new double[N];
        memset(arr[i], 0, N * sizeof(double));
        memset(arr2[i], 0, N * sizeof(double));
    }
    
    double delta = 10.0 / N;
    int real_number_of_iteration = 0;


    arr[0][0] = 10.0, arr[0][N - 1] = 20.0, arr[N - 1][0] = 20.0, arr[N - 1][N - 1] = 30.0;
    arr2[0][0] = 10.0, arr2[0][N - 1] = 20.0, arr2[N - 1][0] = 20.0, arr2[N - 1][N - 1] = 30.0;

#pragma acc parallel loop
    for(int i = 1; i < N - 1; i++){
        arr[i][0] = arr[i - 1][0] + delta;
        arr[i][N - 1] = arr[i - 1][N - 1] + delta;
        arr[0][i] = arr[0][i - 1] + delta;
        arr[N - 1][i] = arr[N - 1][i - 1] + delta;
        arr2[i][0] = arr2[i - 1][0] + delta;
        arr2[i][N - 1] = arr2[i - 1][N - 1] + delta;
        arr2[0][i] = arr2[0][i - 1] + delta;
        arr2[N - 1][i] = arr2[N - 1][i - 1] + delta;
    }

#pragma acc enter data create(delta)
#pragma acc data copy(arr[0:N][0:N]) create(arr2[0:N][0:N])

    for (int t = 0; t < num_of_iter; t++){
        real_number_of_iteration = t + 1;

        delta = 0.0;
#pragma acc update device (delta)        
#pragma acc parallel loop reduction(max:delta)
        for (int i = 1; i < N - 1; i++){
#pragma acc loop reduction(max:delta)
            for (int j = 1; j < N - 1; j++){
                arr2[i][j] = (arr[i - 1][j] + arr[i][j-1] + arr[i + 1][j] + arr[i][j + 1]) * 0.25;
                if (arr2[i][j] - arr[i][j] > delta) delta = arr2[i][j] - arr[i][j];
            }
        }

#pragma acc parallel loop
        for (int i = 1; i < N - 1; i++){
#pragma acc loop
            for (int j = 0; j < N - 1; j++){
                arr[i][j] = arr2[i][j];
            }
        }
        
#pragma acc update self (delta)
        if (delta < accuracy) break;
    }

#pragma acc exit data delete(arr[0:N][0:N],arr2[0:N][0:N]) copyout(delta)
    std::cout << "Number of iteration: " << real_number_of_iteration << "\nAccuracy: " << delta << std::endl;

    auto elapsed = std::chrono::high_resolution_clock::now() - start;
	long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	printf("%lld\n", microseconds);

    //delete[] arr;

    return 0;
}