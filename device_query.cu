#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


int main()
{
	int device_count = 0;
	cudaGetDeviceCount(&device_count);
	printf("gpu count: %d\n", device_count);

	cudaDeviceProp device_prop;

	for (int i = 0; i < device_count; i++)
	{
		cudaGetDeviceProperties(&device_prop, i);

		printf("\n\n\n");
		printf("gpu model: %s\n", device_prop.name);
		printf("gpu memory capacity(MB): %f\n",float(device_prop.totalGlobalMem) /(1024.0*1024.0));
	}

	return 0;
}
