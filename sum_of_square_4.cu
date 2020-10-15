#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <device_launch_parameters.h>


#define DATA_SIZE 99999
#define BLOCK_NUM 1

void generate_numbers(int* number, int size)
{
	for (int i = 0;i < size;i++)
	{
		number[i] = rand() % 10;
	}
}


//__global__ static void sum_of_square(int *num, int *result)
//{
//	int x = threadIdx.x;
//	int y = threadIdx.y;
//
//	int sum = 0;
//	for (int i = x * 8 + y; i < DATA_SIZE; i += 256)
//	{
//		sum += num[i] * num[i];
//	}
//
//	result[x * 8 + y] = sum;
//}

__global__ static void sum_of_square(int *num, int *result)
{
	int x = threadIdx.x;
	int y = threadIdx.y;

	int sum = 0;
	for (int i = y * 32 + x; i < DATA_SIZE; i += 256)
	{
		sum += num[i] * num[i];
	}

	result[y * 32 + x] = sum;
}


int main()
{
	int* data = (int*)malloc(DATA_SIZE * sizeof(int));

	generate_numbers(data, DATA_SIZE);

	int* gpu_data, * result;

	dim3 thread_per_block(32, 8);

	cudaMalloc((void**)&gpu_data, DATA_SIZE * sizeof(int));
	cudaMalloc((void**)&result, BLOCK_NUM * thread_per_block.x * thread_per_block.y * sizeof(int));
	
	cudaMemcpy(gpu_data, data, DATA_SIZE * sizeof(int), cudaMemcpyHostToDevice);

	sum_of_square << <BLOCK_NUM, thread_per_block, 0 >> > (gpu_data, result);

	int* sum = (int*)malloc(thread_per_block.x * thread_per_block.y * BLOCK_NUM * sizeof(int));
	cudaMemcpy(sum, result, sizeof(int) * thread_per_block.x * thread_per_block.y * BLOCK_NUM, cudaMemcpyDeviceToHost);

	cudaFree(gpu_data);
	cudaFree(result);

	int sum_gpu = 0;
	for (int i = 0;i < thread_per_block.x * thread_per_block.y * BLOCK_NUM;i++)
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