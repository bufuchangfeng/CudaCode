#include <stdio.h>
#include <cuda_runtime.h>

__global__ void hello_world_from_gpu(void)
{
	printf("Hello World from GPU\n");
	return;
}


int main(void)
{
	printf("Hello World from CPU\n");

	hello_world_from_gpu <<<1, 1>>> ();

	cudaDeviceReset();
	
	return 0;
}

