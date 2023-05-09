#include <cstdio>
#include <cstdlib>

__global__ void bucket_sort(int* key, int range) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  extern __shared__ int bucket[];
  atomicAdd(&bucket[key[i]], 1);
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int j=0, k=0; j<range; j++) {
      for (; bucket[j]>0; bucket[j]--) {
        key[k++] = j;
      }
    }
  }
}

int main() {
  int n = 50;
  int range = 5;
  int* key;
  cudaMallocManaged(&key, n*sizeof(int));
  
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  bucket_sort<<<1,n,range>>>(key, range);
  cudaDeviceSynchronize();

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
}
