#include <stdio.h>
#include <malloc.h>
#include <math.h>
#include <chrono>

int n = 10000000;
double pi = 3.14159265359;	

int main() {
	auto start = std::chrono::high_resolution_clock::now();
	double* arr = (double*)malloc(n * sizeof(double));
	double sum = 0;

#pragma acc enter data create(arr[0:n],sum)

#pragma acc parallel loop present(arr[0:n])

	for (int i = 0; i < n; i++) {
		arr[i] = sin(2 * pi * i / n);
	}

#pragma acc parallel loop present(arr[0:n],sum) reduction(+:sum)
	for (int i = 0; i < n; i++) sum += arr[i];
#pragma acc exit data delete(arr[0:n]) copyout(sum)
	printf("%0.30lf\n", sum);

	auto elapsed = std::chrono::high_resolution_clock::now() - start;
	long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	printf("%lld\n", microseconds);

	free(arr);
	return 0;
}
