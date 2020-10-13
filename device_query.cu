#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


int main()
{
	int device_count = 0;
	cudaGetDeviceCount(&device_count);

	printf("共有 %d 张显卡\n", device_count);

	cudaDeviceProp device_prop;

	for (int i = 0; i < device_count; i++)
	{
		cudaGetDeviceProperties(&device_prop, i);

		printf("\n\n\n");
		printf("显卡型号: %s\n", device_prop.name);
		printf("显卡全局内存容量(MB): %f\n",float(device_prop.totalGlobalMem) /(1024.0*1024.0));
	}

	return 0;
}