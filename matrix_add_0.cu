#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define N 5

__global__ void matrix_add(int* A, int* B, int* C)
{
	// 这里需要注意，cuda的xy 和 数组的xy 是相反的
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	C[i * N + j] = A[i * N + j] + B[i * N + j];
}


void print_matrix(int *M)
{
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			printf("%d ", M[i * N + j]);
		}
		printf("\n");
	}
}


void check_cuda_error(cudaError_t status)
{
	if (status != cudaSuccess)
	{
		printf("error: %s\n", cudaGetErrorString(status));
		exit(1);
	}
}

int main()
{
	int n_blocks = 1;
	dim3 threads_per_block(N, N);


	int* A = (int*)malloc(N * N * sizeof(int));
	int* B = (int*)malloc(N * N * sizeof(int));
	int* C = (int*)malloc(N * N * sizeof(int));
	

	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			A[i * N + j] = B[i * N + j] = i - j;
		}
	}

	int* cuda_A;
	int* cuda_B;
	int* cuda_C;
	
	cudaError_t cuda_status;

	cuda_status = cudaMalloc((void**)&cuda_A, N * N * sizeof(int));
	check_cuda_error(cuda_status);

	cuda_status = cudaMalloc((void**)&cuda_B, N * N * sizeof(int));
	check_cuda_error(cuda_status);

	cuda_status = cudaMalloc((void**)&cuda_C, N * N * sizeof(int));
	check_cuda_error(cuda_status);


	cudaMemcpy(cuda_A, A, N * N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_B, B, N * N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_C, C, N * N * sizeof(int), cudaMemcpyHostToDevice);

	matrix_add << <n_blocks, threads_per_block >> > (cuda_A, cuda_B, cuda_C);

	cudaMemcpy(C, cuda_C, N * N * sizeof(int), cudaMemcpyDeviceToHost);

	printf("A\n");
	print_matrix(A);

	printf("B\n");
	print_matrix(B);

	printf("C\n");
	print_matrix(C);

	cudaFree(cuda_A);
	cudaFree(cuda_B);
	cudaFree(cuda_C);
	free(A);
	free(B);
	free(C);
	return 0;
}