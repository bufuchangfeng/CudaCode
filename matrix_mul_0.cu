#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <math.h>


#define M 3 
#define N 4
#define P 5


__global__ void matrix_mul(int* A, int* B, int* C)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	if (i < M && j < P)
	{
		int sum = 0;
		for (int k = 0; k < N;k++)
		{
			sum += A[i * N + k] * B[k * P + j];
		}
		//printf("%d %d %d\n", i, j, sum);
		C[i * P + j] = sum;
	}
}

void print_matrix(int* matrix, int rows, int cols)
{
	for (int i = 0; i < rows;i++)
	{
		for (int j = 0;j < cols;j++)
		{
			printf("%d ", matrix[i * cols + j]);
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

	int* A;
	int* B;
	int* C;

	A = (int*)malloc(M * N * sizeof(int));
	B = (int*)malloc(N * P * sizeof(int));
	C = (int*)malloc(M * P * sizeof(int));

	for (int i = 0;i < M;i++)
	{
		for (int j = 0;j < N;j++)
		{
			A[i * N + j] = i - j;
		}
	}

	for (int i = 0;i < N;i++)
	{
		for (int j = 0; j < P; j++)
		{
			B[i * P + j] = i + j;
		}
	}

	int* cuda_A;
	int* cuda_B;
	int* cuda_C;

	cudaError_t cuda_status;

	cuda_status = cudaMalloc((void**)&cuda_A, M * N * sizeof(int));
	check_cuda_error(cuda_status);

	cuda_status = cudaMalloc((void**)&cuda_B, N * P * sizeof(int));
	check_cuda_error(cuda_status);

	cuda_status = cudaMalloc((void**)&cuda_C, M * P * sizeof(int));
	check_cuda_error(cuda_status);

	cudaMemcpy(cuda_A, A, M * N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_B, B, N * P * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_C, C, M * P * sizeof(int), cudaMemcpyHostToDevice);

	dim3 threads_per_block(2, 3);

	int n_blocks_x = (int)ceil(float(M) / threads_per_block.x);
	int n_blocks_y = (int)ceil(float(P) / threads_per_block.y);

	dim3 n_blocks(n_blocks_x, n_blocks_y);

	matrix_mul << <n_blocks, threads_per_block >> > (cuda_A, cuda_B, cuda_C);

	cudaMemcpy(C, cuda_C, M * P * sizeof(int), cudaMemcpyDeviceToHost);

	printf("A\n");
	print_matrix(A, M, N);

	printf("B\n");
	print_matrix(B, N, P);

	printf("C\n");
	print_matrix(C, M, P);

	cudaFree(cuda_A);
	cudaFree(cuda_B);
	cudaFree(cuda_C);

	free(A);
	free(B);
	free(C);

	return 0;
}