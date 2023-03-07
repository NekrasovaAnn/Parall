#include <iostream>
#include <cstring>
#include <chrono>

int main() {
    auto start = std::chrono::high_resolution_clock::now();
    const int N = 1024;
    double accuracy = 0.000001;
    int num_of_iter = 1000000;

    double arr[N][N] = {0.0};
    double arr2[N][N] = {0.0};
    
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
#pragma acc parallel loop collapse(2) reduction(max:delta)
        for (int i = 1; i < N - 1; i++){
            for (int j = 1; j < N - 1; j++){
                arr2[i][j] = (arr[i - 1][j] + arr[i][j-1] + arr[i + 1][j] + arr[i][j + 1]) * 0.25;
                if (arr2[i][j] - arr[i][j] > delta) delta = arr2[i][j] - arr[i][j];
            }
        }

#pragma acc parallel loop collapse(2)
        for (int i = 1; i < N - 1; i++){
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

    return 0;
}