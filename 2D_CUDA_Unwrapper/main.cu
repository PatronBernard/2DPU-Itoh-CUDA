#include "Functions.h"
#include <iostream>
//Profiling
#include <time.h>

//Fixt intellisense problemen
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

__global__ void unwrap_gpu(float* d_Output, float* d_image, int image_edge,float cutoff){
	//Elke thread krijgt 1 kolom
		int column=(threadIdx.x)+blockDim.x*(blockIdx.x);
		if(column < image_edge){
			float deltaref=0;
			d_Output[column*image_edge]=d_image[column*image_edge];
			for(int row=0;row<image_edge-1;row++){
				float delta=d_image[column*image_edge+row+1]-d_image[column*image_edge+row];
				if(fabsf(delta)>=cutoff)
					deltaref=deltaref-delta;
				d_Output[column*image_edge+row+1]=d_image[column*image_edge+row+1]+deltaref;
			}
		}
		//Other direction
		if(column <= image_edge){
			float deltaref=0;
			for(int row=0;row<=image_edge;row++){
				float delta=d_Output[column+row*image_edge+image_edge]-d_Output[column+row*image_edge];
				if(fabsf(delta)>=cutoff)
					deltaref=deltaref-delta;
				d_Output[column+row*image_edge]=d_Output[column+row*image_edge+image_edge]+deltaref;
			}
		}
}

int main ()
{
	//Useful constants
	int image_size=768;
	int elements_amount=(int) pow((float)image_size,2);
	int sample_size=100;


	//Read the binary file containing the wrapped & flattened image
	float* Data;
	Data=read_data("C:\\Users\\janmorez\\Dropbox\\MATLAB\\CUDA\\Image_Wrapped.bin");

	//=============CPU-unwrap=============
	clock_t begin,end;

	if(image_size < 1024){
		float* CPUTimeArray=(float*)malloc(sizeof(float)*sample_size);
		float * OutputCPU;
		for(int i=0;i<sample_size;i++){
			begin=clock();
			OutputCPU=unwrap_cpu(Data,image_size);
			end=clock();
			float CPUTime=1000*(float)(end-begin)/CLOCKS_PER_SEC;
			CPUTimeArray[i]=CPUTime;
			printf("CPU Unwrap time was: %f ms \n",CPUTime);
		}
		export_data("C:\\Users\\janmorez\\Dropbox\\MATLAB\\CUDA\\OutputCPU.bin",OutputCPU,elements_amount);
		export_data("C:\\Users\\janmorez\\Dropbox\\MATLAB\\CUDA\\OutputCPUTimes.bin",CPUTimeArray,sample_size);

		free(OutputCPU);
		free(CPUTimeArray);
		}


	//=============GPU-unwrap=============
	//Determine gridsize & dimensions
	printf("\n =============================\n Starting GPU-Unwrap \n");
	cudaDeviceProp device;
	cudaGetDeviceProperties(&device,0);
	int MaxThreadsPerBlock=device.maxThreadsPerBlock;
	dim3 BlockDim;
	BlockDim.x=MaxThreadsPerBlock;
	int GridSize=ceil((float)image_size/MaxThreadsPerBlock);
	printf("Amount of threads per block: %i. Gridsize (in blocks): %i \n",MaxThreadsPerBlock,GridSize);

	//Profiling code
	cudaEvent_t startG,stopG;
	cudaEventCreate(&startG);
	cudaEventCreate(&stopG);
	float* GPUTimeArray=(float*)malloc((size_t) sizeof(float)*sample_size);
	float timeG;
	float* h_OutputGPU;

	for(int i=0;i< sample_size;i++){
	//Allocate device Output memory
	cudaError err=cudaSuccess;
	float* d_Output=NULL;
	err=cudaMalloc(&d_Output,elements_amount*sizeof(float));
	if(err==cudaSuccess){
		printf("Starting GPU-unwrap... \n");
		cudaEventRecord(startG,0);
		float* d_Data=copy_data_to_device(Data,image_size);
		unwrap_gpu<<<GridSize,BlockDim>>>(d_Output,d_Data,image_size,1);
		cudaThreadSynchronize();
		cudaDeviceSynchronize();
		h_OutputGPU=copy_data_to_host(d_Output,image_size);	
		cudaEventRecord(stopG);
		cudaFree(d_Output);
		cudaFree(d_Data);
		printf("GPU-unwrap completed. Possible kernel errors: %s. \n",cudaGetErrorString(cudaGetLastError()));

	}
	else
		printf("GPU memory allocation failed: %s \n",cudaGetErrorString(err));


	cudaEventSynchronize(stopG);
	cudaEventElapsedTime(&timeG,startG,stopG);
	GPUTimeArray[i]=timeG;
	printf("Complete GPU-unwrap took: %f ms \n",GPUTimeArray[i]);
	}
	

	export_data("C:\\Users\\janmorez\\Dropbox\\MATLAB\\CUDA\\OutputGPU.bin",h_OutputGPU,elements_amount);
	export_data("C:\\Users\\janmorez\\Dropbox\\MATLAB\\CUDA\\OutputGPUTimes.bin",GPUTimeArray,sample_size);

	free(Data);
	free(GPUTimeArray);
	//=============END OF PROGRAM=========
	printf("Program ended. Press any key to close this window... \n ");
	getchar(); 
	return 0;
	
}