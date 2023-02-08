#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>


# define M 5
# define N 4

__global__ void matrix_add(int* A, int* B, int* C)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;
	
	if (i < M && j < N) {
		C[i * N + j] = A[i * N + j] + B[i * N + j];
	}
}

void print_matrix(int* matrix)
{
	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < N; j++)
		{
			printf("%d ", matrix[i * N + j]);
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
	dim3 threads_per_block(2, 3);

	int n_blocks_x = (int)ceil(float(M) / threads_per_block.x);
	int n_blocks_y = (int)ceil(float(N) / threads_per_block.y);

	dim3 n_blocks(n_blocks_x, n_blocks_y);

	int* A = (int*)malloc(M * N * sizeof(int));
	int* B = (int*)malloc(M * N * sizeof(int));
	int* C = (int*)malloc(M * N * sizeof(int));

	for (int i = 0; i < M;i++)
	{
		for (int j = 0; j < N;j++)
		{
			A[i * N + j] = B[i * N + j] = rand() % 10;
		}
	}

	int* cuda_A;
	int* cuda_B;
	int* cuda_C;

	cudaError_t cuda_status;

	cuda_status = cudaMalloc((void**)&cuda_A, M * N * sizeof(int));
	check_cuda_error(cuda_status);

	cuda_status = cudaMalloc((void**)&cuda_B, M * N * sizeof(int));
	check_cuda_error(cuda_status);

	cuda_status = cudaMalloc((void**)&cuda_C, M * N * sizeof(int));
	check_cuda_error(cuda_status);

	cudaMemcpy(cuda_A, A, M * N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_B, B, M * N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_C, C, M * N * sizeof(int), cudaMemcpyHostToDevice);
	
	matrix_add << <n_blocks, threads_per_block >> > (cuda_A, cuda_B, cuda_C);

	cudaMemcpy(C, cuda_C, M * N * sizeof(int), cudaMemcpyDeviceToHost);

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
