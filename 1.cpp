#include <stdio.h>
#include <malloc.h>
#include <math.h>

int n = 10000000;
double pi = 3.14159265359;	

int main() {
	double* arr = (double*)malloc(n * sizeof(double));
	double sum = 0;

#pragma acc enter data create(array[0:n],sum)

#pragma acc parallel loop present(array[0:n])

	for (int i = 0; i < n; i++) {
		arr[i] = sin(2 * pi * i / n);
	}

#pragma acc parallel loop present(array[0:n],sum) reduction(+:sum)
	for (int i = 0; i < n; i++) sum += arr[i];
#pragma acc exit data delete(array[0:n]) copyout(sum)
	printf("%0.30lf\n", sum);

	free(arr);
	return 0;
}