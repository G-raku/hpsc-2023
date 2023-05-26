#include <cstdio>
#include <cstdlib>

__device__ void align(int* key, int offset, int quantity, int n) {
  for (int j=0; j<quantity; j++) {
    key[offset+j] = n;
  }
}

__global__ void bucket_sort(int* key, int range) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  extern __shared__ int bucket[];
  atomicAdd(&bucket[key[i]], 1);
  __syncthreads();
  if (threadIdx.x == 0) {
    int offset = 0;
    for (int j=0; j<range; j++) {
      align(key, offset, bucket[j], j);
      offset += bucket[j];
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
