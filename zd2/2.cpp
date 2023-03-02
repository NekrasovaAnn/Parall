#include <iostream>
#include <cstring>

double accuracy = 0.000001;
int num_of_iter = 1000000;

int main() {
    const int N = 128;

    double arr[N][N] = { 0.0 };
    double arr2[N][N] = { 0.0 };

    double delta = 10.0 / N;
    int real_number_of_iteration = 0;

    #pragma acc enter data create(arr[0:N],delta,real_number_of_iteration)
    #pragma acc data copy(arr[1:N][1:N]) create(arr2[N][N])

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

    for (int t = 0; t < num_of_iter; t++){
        real_number_of_iteration = t + 1;

        delta = 0.0;
        #pragma acc parallel loop reduction(max:delta)
        for (int i = 1; i < N - 1; i++){
            #pragma acc loop reduction(max:delta)
            for (int j = 1; j < N - 1; j++){
                arr2[i][j] = (arr[i - 1][j] + arr[i][j-1] + arr[i + 1][j] + arr[i][j + 1]) * 0.25;
                if (arr2[i][j] - arr[i][j] > delta) delta = arr2[i][j] - arr[i][j];
            }
        }

        //std::memcpy(arr, arr2, N * N * sizeof(double));
        #pragma acc parallel loop
        for (int i = 1; i < N - 1; i++){
            #pragma acc loop
            for (int j = 0; j < N - 1; j++){
                arr[i][j] = arr2[i][j];
            }
        }

        if (delta < accuracy) break;
    }

#pragma acc exit data delete(arr[0:N][0:N],arr2[0:N][0:N]) copyout(real_number_of_iteration,delta)
    std::cout << "Number of iteration: " << real_number_of_iteration << "\nAccuracy: " << delta << std::endl;

    return 0;
}