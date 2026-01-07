#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <algorithm>

#define MAX_THREADS_PER_BLOCK 1024
// #define N_ELEM 1048576
// #define N_ELEM 2000000
#define N_ELEM 50
#define VERY_LARGE_NUMBER 99999

// Bitonic Sort for CPU
void bitonicSortCPU(int* arr, int n) 
{
    for (int k = 2; k <= n; k *= 2) 
    {
        for (int j = k / 2; j > 0; j /= 2) 
        {
            for (int i = 0; i < n; i++) 
            {
                int ij = i ^ j;

                if (ij > i) 
                {
                    if ((i & k) == 0) 
                    {
                        if (arr[i] > arr[ij])
                        {
                            int temp = arr[i];
                            arr[i] = arr[ij];
                            arr[ij] = temp;
                        }
                    }
                    else 
                    {
                        if (arr[i] < arr[ij])
                        {
                            int temp = arr[i];
                            arr[i] = arr[ij];
                            arr[ij] = temp;
                        }
                    }
                }
            }
        }
    }
}

//GPU Kernel Implementation of Bitonic Sort
__global__ void bitonicSortGPU(int* arr, int j, int k)
{
    unsigned int i, ij;

    i = threadIdx.x + blockDim.x * blockIdx.x;

    ij = i ^ j;

    if (ij > i)
    {
        if ((i & k) == 0)
        {
            if (arr[i] > arr[ij])
            {
                int temp = arr[i];
                arr[i] = arr[ij];
                arr[ij] = temp;
            }
        }
        else
        {
            if (arr[i] < arr[ij])
            {
                int temp = arr[i];
                arr[i] = arr[ij];
                arr[ij] = temp;
            }
        }
    }
}

//Function to print array
void printArray(int* arr, int size) 
{
    for (int i = 0; i < size; ++i)
        std::cout << arr[i] << " ";
    std::cout << std::endl;
}

//Automated function to check if array is sorted
bool isSorted(int* arr, int size) 
{
    for (int i = 1; i < size; ++i) 
    {
        if (arr[i] < arr[i - 1])
            return false;
    }
    return true;
}

unsigned int next_power_of_2(unsigned int n) {
    // https://graphics.stanford.edu/%7Eseander/bithacks.html#RoundUpPowerOf2
    // The code works for 32 bit data types
    if (n == 0) return 1;
    
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
}

__global__ void copy_with_pad(int *src, int *dst, unsigned int arr_size, unsigned int next_pow){
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    while (idx < next_pow){
        if (idx < arr_size){
            dst[idx] = src[idx];
        }
        else {
            dst[idx] = VERY_LARGE_NUMBER;
        }

        idx += blockDim.x * gridDim.x;
    }
}

__global__ void copy_back_from_padded_array(int *src, int *dst, unsigned int arr_size){
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < arr_size){
        dst[idx] = src[idx];
        idx += blockDim.x * gridDim.x;
    }
}


//MAIN PROGRAM
int main(){   
    unsigned int size = N_ELEM;
    //Create CPU based Arrays
    int* arr = new int[size];
    int* carr = new int[size];
    int* temp = new int[size];

    //Create GPU based arrays
    int* gpuArrbiton;
    int* gpuArrbiton_2; // Pads the size to the nearest power of 2
    int* gpuTemp;

    // Initialize the array with random values
    srand(static_cast<unsigned int>(time(nullptr)));
    for (int i = 0; i < size; ++i) 
    {
        arr[i] = rand() % 100;
        carr[i] = arr[i];
    }

    //Print unsorted array 
    std::cout << "\n\nUnsorted array: ";
    if (size <= 100) 
    {
        printArray(arr, size);
    }
    else 
    {
        printf("\nToo Big to print. Check Variable. Automated isSorted Checker will be implemented\n");
    }

    cudaMalloc((void**)&gpuTemp, size * sizeof(int));
    cudaMalloc((void**)&gpuArrbiton, size * sizeof(int));
    
    
    cudaMemcpy(gpuArrbiton, arr, size * sizeof(int), cudaMemcpyHostToDevice);
    
    // Perform GPU merge sort and measure time
    cudaEvent_t startGPU, stopGPU;
    cudaEventCreate(&startGPU);
    cudaEventCreate(&stopGPU);
    float millisecondsGPU = 0;

    //Initialize CPU clock counters
    clock_t startCPU, endCPU;

    //Set number of threads and blocks for kernel calls
    int threadsPerBlock = MAX_THREADS_PER_BLOCK;
    int blocksPerGrid;
    // blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    int j, k;

    unsigned int next_pow = next_power_of_2(size);
    cudaMalloc((void**)&gpuArrbiton_2, next_pow * sizeof(int));

    //Time the run and call GPU Bitonic Kernel
    cudaEventRecord(startGPU);
    blocksPerGrid = (next_pow + threadsPerBlock - 1) / threadsPerBlock;
    copy_with_pad <<<blocksPerGrid, threadsPerBlock>>>(gpuArrbiton, gpuArrbiton_2, size, next_pow);
    // for (k = 2; k <= size; k <<= 1)
    for (k = 2; k <= next_pow; k <<= 1)
    {
        for (j = k >> 1; j > 0; j = j >> 1)
        {
            // bitonicSortGPU << <blocksPerGrid, threadsPerBlock >> > (gpuArrbiton, j, k);
            bitonicSortGPU << <blocksPerGrid, threadsPerBlock >> > (gpuArrbiton_2, j, k);
        }
    }
    blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    copy_back_from_padded_array<< <blocksPerGrid, threadsPerBlock >> > (gpuArrbiton_2, gpuArrbiton, size);
    cudaEventRecord(stopGPU);

    //Transfer Sorted array back to CPU
    cudaMemcpy(arr, gpuArrbiton, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stopGPU);
    cudaEventElapsedTime(&millisecondsGPU, startGPU, stopGPU);

    //Time the run and call CPU Bitonic Sort
    // startCPU = clock();
    // bitonicSortCPU(carr, size);
    // endCPU = clock();

    // //Calculate Elapsed CPU time
    // double millisecondsCPU = static_cast<double>(endCPU - startCPU) / (CLOCKS_PER_SEC / 1000.0);

    // Display sorted GPU array
    std::cout << "\n\nSorted GPU array: ";
    if (size <= 100) 
    {
        printArray(arr, size);
    }
    else {
        printf("\nToo Big to print. Check Variable. Automated isSorted Checker will be implemented\n");
    }

    //Display sorted CPU array
    // std::cout << "\nSorted CPU array: ";
    // if (size <= 100) 
    // {
    //     printArray(carr, size);
    // }
    // else {
    //     printf("\nToo Big to print. Check Variable. Automated isSorted Checker will be implemented\n");
    // }
    
    //Run the array with the automated isSorted checker
    if (isSorted(arr, size))
        std::cout << "\n\nSORT CHECKER RUNNING - SUCCESFULLY SORTED GPU ARRAY" << std::endl;
    else
        std::cout << "SORT CHECKER RUNNING - !!! FAIL !!!" << std::endl;
   
    // if (isSorted(carr, size))
    //     std::cout << "SORT CHECKER RUNNING - SUCCESFULLY SORTED CPU ARRAY" << std::endl;
    // else
    //     std::cout << "SORT CHECKER RUNNING - !!! FAIL !!!" << std::endl;

    //Print the time of the runs
    std::cout << "\n\nGPU Time: " << millisecondsGPU << " ms" << std::endl;
    // std::cout << "CPU Time: " << millisecondsCPU << " ms" << std::endl;

    //Destroy all variables
    delete[] arr;
    delete[] carr;
    delete[] temp;

    //End
    // cudaFree(gpuArrmerge);
    cudaFree(gpuArrbiton);
    cudaFree(gpuArrbiton_2);
    cudaFree(gpuTemp);

    // unsigned int arr_size = 54;
    // unsigned int next_pow = next_power_of_2(arr_size);
    // printf("Given array size: %d, next power of 2 is: %d\n", arr_size, next_pow);

}