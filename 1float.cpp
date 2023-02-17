#include <stdio.h>
#include <malloc.h>
#include <math.h>
#include <chrono>

int n = 10000000;
float pi = 3.14159265359;	

int main() {
	float* arr = (float*)malloc(n * sizeof(double));
	float sum = 0;

#pragma acc enter data create(arr[0:n],sum)

#pragma acc parallel loop present(arr[0:n])

	for (int i = 0; i < n; i++) {
		arr[i] = sin(2 * pi * i / n);
	}

#pragma acc parallel loop present(arr[0:n],sum) reduction(+:sum)
	for (int i = 0; i < n; i++) sum += arr[i];
#pragma acc exit data delete(arr[0:n]) copyout(sum)
	printf("%0.30f\n", sum);

	free(arr);
	return 0;
}
