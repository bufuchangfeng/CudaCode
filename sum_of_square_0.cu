#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define DATA_SIZE 99999

void generate_numbers(int* number, int size)
{
	for (int i = 0;i < size;i++)
	{
		number[i] = rand() % 10;
	}
}


__global__ static void sum_of_square(int *num, int *result)
{
	int sum = 0;
	for (int i = 0; i < DATA_SIZE; i++)
	{
		sum += num[i] * num[i];
	}

	*result = sum;
}


int main()
{
	int* data = (int*)malloc(DATA_SIZE * sizeof(int));

	generate_numbers(data, DATA_SIZE);

	int* gpu_data, * result;

	cudaMalloc((void**)&gpu_data, DATA_SIZE * sizeof(int));
	cudaMalloc((void**)&result, sizeof(int));
	
	cudaMemcpy(gpu_data, data, DATA_SIZE * sizeof(int), cudaMemcpyHostToDevice);

	sum_of_square << <1, 1, 0 >> > (gpu_data, result);

	int sum_gpu;
	cudaMemcpy(&sum_gpu, result, sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(gpu_data);
	cudaFree(result);

	int sum_cpu = 0;
	for (int i = 0; i < DATA_SIZE; i++)
	{
		sum_cpu += data[i] * data[i];
	}

	if (sum_cpu == sum_gpu)
	{
		printf("True\n");
	}
	else
	{
		printf("False\n");
	}

	free(data);

	return 0;
}