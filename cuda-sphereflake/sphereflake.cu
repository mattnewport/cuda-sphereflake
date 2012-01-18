#include <stdio.h>
#include <cuda.h>

__global__ void sphereflake()
{
	printf("blockIdx is (%d, %d), threadIdx is: (%d, %d, %d)\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, threadIdx.z);
}

int main(int argc, char **argv)
{
	dim3 dimGrid(2, 2);
	dim3 dimBlock(2, 2, 2);
	sphereflake<<<dimGrid, dimBlock>>>();
	
	cudaDeviceSynchronize();

    cudaDeviceReset();

	return 0;
}

