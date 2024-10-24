#include <stdio.h>
#include <assert.h>

// Or we can wrap the checkCuda function, which returns a type cudaError_t
inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) 
  {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

void init(int *a, int N)
{
  int i;
  for (i = 0; i < N; ++i)
  {
    a[i] = i;
  }
}

__global__ void doubleElements(int *a, int N)
{

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;

  for (int i = idx; i < N + stride; i += stride)
  {
    a[i] *= 2;
  }
}

bool checkElementsAreDoubled(int *a, int N)
{
  int i;
  for (i = 0; i < N; ++i)
  {
    if (a[i] != i*2) return false;
  }
  return true;
}

int main()
{
  /*
   * Add error handling to this source code to learn what errors
   * exist, and then correct them. Googling error messages may be
   * of service if actions for resolving them are not clear to you.
   */

  int N = 10000;
  int *a;
  cudaError_t syncErr, asyncErr;

  size_t size = N * sizeof(int);
  cudaMallocManaged(&a, size);

  init(a, N);

  size_t threads_per_block = 2048; // -> max threads_per_block is 1024!
  size_t number_of_blocks = 32;

  doubleElements<<<number_of_blocks, threads_per_block>>>(a, N);

  /*
   * Catch errors for both the kernel launch above and any
   * errors that occur during the asynchronous `doubleElements`
   * kernel execution.
   */
  syncErr = cudaGetLastError();
  asyncErr = cudaDeviceSynchronize();
  

  if (syncErr != cudaSuccess)
  {  
    // Error in the kernel execution
    printf("Sync Error: %s\n", cudaGetErrorString(syncErr));
  }
  if (asyncErr != cudaSuccess) 
  {
    printf("Async Error: %s\n", cudaGetErrorString(asyncErr));
  }

   checkCuda( cudaDeviceSynchronize() );
  bool areDoubled = checkElementsAreDoubled(a, N);
  printf("All elements were doubled? %s\n", areDoubled ? "TRUE" : "FALSE");

  cudaFree(a);
}
