#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>


__global__ void vecor_add(int* A, int* B, int* C, int N)
{
	int i = threadIdx.x;

	C[i] = A[i] + B[i];
}

int main()
{
	int N = 8;
	int* A = (int*)malloc(N * sizeof(int));
	int* B = (int*)malloc(N * sizeof(int));
	int* C = (int*)malloc(N * sizeof(int));

	int* cuda_A, *cuda_B, *cuda_C;

	cudaMalloc((void**)&cuda_A, N * sizeof(int));
	cudaMalloc((void**)&cuda_B, N * sizeof(int));
	cudaMalloc((void**)&cuda_C, N * sizeof(int));

	for (int i = 0;i < N;i++)
	{
		A[i] = i - 1;
		B[i] = i + 1;

		//printf("%d %d\n", A[i], B[i]);
	}

	cudaError_t cuda_status;

	cuda_status = cudaMemcpy(cuda_A, A, N * sizeof(int), cudaMemcpyHostToDevice);
	
	if (cuda_status != cudaSuccess)
	{
		printf("error: %s\n", cudaGetErrorString(cuda_status));
		return 1;
	}

	cuda_status = cudaMemcpy(cuda_B, B, N * sizeof(int), cudaMemcpyHostToDevice);

	if (cuda_status != cudaSuccess)
	{
		printf("error: %s\n", cudaGetErrorString(cuda_status));
		return 1;
	}

	vecor_add << <1, N >> > (cuda_A, cuda_B, cuda_C, N);

	cuda_status = cudaMemcpy(C, cuda_C, N * sizeof(int), cudaMemcpyDeviceToHost);
	
	if (cuda_status != cudaSuccess)
	{
		printf("error: %s\n", cudaGetErrorString(cuda_status));
		return 1;
	}

	for (int i = 0;i < N;i++)
	{
		printf("%d + %d = %d\n", A[i], B[i], C[i]);
	}

	cudaFree(cuda_A);
	cudaFree(cuda_B);
	cudaFree(cuda_C);

	free(A);
	free(B);
	free(C);

	return 0;
}