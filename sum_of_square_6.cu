#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>
#include <cuda.h>


#define DATA_SIZE 99999
#define THREAD_NUM 256
#define BLOCK_NUM 32

void generate_numbers(int* number, int size)
{
	for (int i = 0;i < size;i++)
	{
		number[i] = rand() % 10;
	}
}


__global__ static void sum_of_square(int *num, int *result)
{
	extern __shared__ int shared[];


	int bid = blockIdx.x;
	int tid = threadIdx.x;

	shared[tid] = 0;
	for (int i = bid * THREAD_NUM + tid; i < DATA_SIZE; i += BLOCK_NUM * THREAD_NUM)
	{
		shared[tid] += num[i] * num[i];
	}

	__syncthreads();
	
	int offset = 1;
	int mask = 1;


	// just a copy
	while(offset < THREAD_NUM) 
	{ 
		if((tid & mask) == 0) 
		{ 
			shared[tid] += shared[tid + offset]; 
		} 
		offset += offset; 
		mask = offset + mask; 
		__syncthreads(); 
	}

	if (tid == 0)
	{
		result[bid] = shared[0];
	}
}


int main()
{

	int* data = (int*)malloc(DATA_SIZE * sizeof(int));

	generate_numbers(data, DATA_SIZE);

	int* gpu_data, * result;

	cudaMalloc((void**)&gpu_data, DATA_SIZE * sizeof(int));
	cudaMalloc((void**)&result, BLOCK_NUM * sizeof(int));
	
	cudaMemcpy(gpu_data, data, DATA_SIZE * sizeof(int), cudaMemcpyHostToDevice);

	sum_of_square << <BLOCK_NUM, THREAD_NUM, THREAD_NUM * sizeof(int) >> > (gpu_data, result);

	int sum[BLOCK_NUM];
	cudaMemcpy(sum, result, sizeof(int) * BLOCK_NUM, cudaMemcpyDeviceToHost);

	cudaFree(gpu_data);
	cudaFree(result);

	int sum_gpu = 0;
	for (int i = 0;i < BLOCK_NUM;i++)
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