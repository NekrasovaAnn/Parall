#include <stdio.h>
#include <cmath>
#include <chrono>

int n = 10000000;
float pi = 3.14159265359;	

int main() {
	auto start = std::chrono::high_resolution_clock::now();
	float* arr = new float[n];
	float sum = 0;

#pragma acc enter data create(arr[0:n],sum)

#pragma acc parallel loop present(arr[0:n])

	for (int i = 0; i < n; i++) {
		arr[i] = sinf(2 * pi * i / n);
	}

#pragma acc parallel loop present(arr[0:n],sum) reduction(+:sum)
	for (int i = 0; i < n; i++) sum += arr[i];
#pragma acc exit data delete(arr[0:n]) copyout(sum)
	printf("%0.30f\n", sum);

	auto elapsed = std::chrono::high_resolution_clock::now() - start;
	long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	printf("%lld\n", microseconds);

	delete [] arr;
	return 0;
}
