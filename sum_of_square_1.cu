#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <device_launch_parameters.h>


#define DATA_SIZE 99999
#define THREAD_NUM 256


void generate_numbers(int* number, int size)
{
	for (int i = 0;i < size;i++)
	{
		number[i] = rand() % 10;
	}
}


__global__ static void sum_of_square(int *num, int *result)
{
	int x = threadIdx.x;
	int size = (int)ceil(float(DATA_SIZE) / float(THREAD_NUM));

	int sum = 0;

	for (int i = x * size;i < (x + 1) * size;i++)
	{
		if (i < DATA_SIZE)
		{
			sum += num[i] * num[i];
		}
	}

	result[x] = sum;
}


int main()
{
	int* data = (int*)malloc(DATA_SIZE * sizeof(int));

	generate_numbers(data, DATA_SIZE);

	int* gpu_data, * result;

	cudaMalloc((void**)&gpu_data, DATA_SIZE * sizeof(int));
	cudaMalloc((void**)&result, THREAD_NUM * sizeof(int));
	
	cudaMemcpy(gpu_data, data, DATA_SIZE * sizeof(int), cudaMemcpyHostToDevice);

	sum_of_square << <1, THREAD_NUM, 0 >> > (gpu_data, result);

	int sum[THREAD_NUM];
	cudaMemcpy(sum, result, sizeof(int) * THREAD_NUM, cudaMemcpyDeviceToHost);

	cudaFree(gpu_data);
	cudaFree(result);

	int sum_gpu = 0;
	for (int i = 0;i < THREAD_NUM;i++)
	{
		sum_gpu += sum[i];
	}

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