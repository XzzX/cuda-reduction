#include <cuda_runtime.h>

#include <cstddef>
#include <sys/time.h>

#include <iostream>
#include <vector>

void checkError(cudaError_t err)
{
    if (err != cudaSuccess)
    {
        std::cout << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }
}

__global__ void reduce_neighbored_global(int* A, int* B, const int N)
{
   // reduction with global memory

   for (int loop=0; loop < 1000; ++loop)
   {
      const int tid = threadIdx.x;
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      const int gridStride = blockDim.x * gridDim.x;

      int temp =  idx + gridStride;
      while (temp < N)
      {
         A[idx] += A[temp];
         temp += gridStride;
      }

      __syncthreads();

       for (int s = 1; s < blockDim.x; s *= 2)
       {
          if (tid % (2*s) == 0)
          {
             A[idx] += A[idx + s];
          }
          __syncthreads();
       }

       if (tid == 0)
         B[blockIdx.x] = A[idx];
   }
}

__global__ void reduce_neighbored_conflict_divergent(int* A, int* B, const int N)
{
   //reduction with shared memory

   for (int loop=0; loop < 1000; ++loop)
   {
      const int tid = threadIdx.x;
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      const int gridStride = blockDim.x * gridDim.x;

      __shared__ int buf[1024];

      buf[tid] = 0;
      while (idx < N)
      {
         buf[tid] += A[idx];
         idx += gridStride;
      }

      __syncthreads();

       for (int s = 1; s < blockDim.x; s *= 2)
       {
          if (tid % (2*s) == 0)
          {
             buf[tid] += buf[tid + s];
          }
          __syncthreads();
       }

       if (tid == 0)
         B[blockIdx.x] = buf[0];
   }
}

__global__ void reduce_neighbored_conflict_nondivergent(int* A, int* B, const int N)
{
   //remove warp divergence

   for (int loop=0; loop < 1000; ++loop)
   {
      const int tid = threadIdx.x;
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      const int gridStride = blockDim.x * gridDim.x;

      __shared__ int buf[1024];

      buf[tid] = 0;
      while (idx < N)
      {
         buf[tid] += A[idx];
         idx += gridStride;
      }

      __syncthreads();

       for (int s = 1; s < blockDim.x; s *= 2)
       {
          const int index = 2 * s * tid;
          if (index < blockDim.x)
          {
             buf[index] += buf[index + s];
          }
          __syncthreads();
       }

       if (tid == 0)
         B[blockIdx.x] = buf[0];
   }
}

__global__ void reduce_interleaved_noconflict_nondivergent(int* A, int* B, const int N)
{
   // remove bank conflicts

   for (int loop=0; loop < 1000; ++loop)
   {
      const int tid = threadIdx.x;
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      const int gridStride = blockDim.x * gridDim.x;

      __shared__ int buf[1024];

      buf[tid] = 0;
      while (idx < N)
      {
         buf[tid] += A[idx];
         idx += gridStride;
      }

      __syncthreads();

       for (int s = blockDim.x/2; s > 0; s >>= 1)
       {
          if (tid < s)
          {
             buf[tid] += buf[tid + s];
          }
          __syncthreads();
       }

       if (tid == 0)
         B[blockIdx.x] = buf[0];
   }
}

__device__ void unroll(volatile int* buf, int tid)
{
    if (tid < 32)
    {
        buf[tid] += buf[tid + 32];
        buf[tid] += buf[tid + 16];
        buf[tid] += buf[tid + 8];
        buf[tid] += buf[tid + 4];
        buf[tid] += buf[tid + 2];
        buf[tid] += buf[tid + 1];
    }
}

__global__ void reduce_interleaved_noconflict_nondivergent_unrolled(int* A, int* B, const int N)
{
   // unroll last loops

   for (int loop=0; loop < 1000; ++loop)
   {
      const int tid = threadIdx.x;
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      const int gridStride = blockDim.x * gridDim.x;

      __shared__ int buf[1024];

      buf[tid] = 0;
      while (idx < N)
      {
         buf[tid] += A[idx];
         idx += gridStride;
      }

      __syncthreads();

       for (int s = blockDim.x/2; s > 32; s >>= 1)
       {
          if (tid < s)
          {
             buf[tid] += buf[tid + s];
          }
          __syncthreads();
       }

       unroll(buf, tid);

       if (tid == 0)
         B[blockIdx.x] = buf[0];
   }
}

template <unsigned int blockSize>
__global__ void reduce_interleaved_noconflict_nondivergent_completelyunrolled(int* A, int* B, const int N)
{
    for (int loop=0; loop < 1000; ++loop)
    {
       const int tid = threadIdx.x;
       int idx = blockIdx.x * blockDim.x + threadIdx.x;
       const int gridStride = blockDim.x * gridDim.x;

       __shared__ int buf[1024];

       buf[tid] = 0;
       while (idx < N)
       {
          buf[tid] += A[idx];
          idx += gridStride;
       }

       __syncthreads();

       if (blockSize>=1024 && tid < 512)
          buf[tid] += buf[tid + 512];
       __syncthreads();

       if (blockSize>=512 && tid < 256)
          buf[tid] += buf[tid + 256];
       __syncthreads();

       if (blockSize>=256 && tid < 128)
          buf[tid] += buf[tid + 128];
       __syncthreads();

       if (blockSize>=128 && tid < 64)
          buf[tid] += buf[tid + 64];
       __syncthreads();

       unroll(buf, tid);

       if (tid == 0)
         B[blockIdx.x] = buf[0];
    }
}

int main()
{
    const int nElem = 1024*2048;
    std::vector<int> A(nElem, 1);
    std::vector<int> B(1024, 1);

    const int nBytes = nElem * sizeof(int);
    std::cout << nBytes * 1e-6 << std::endl;
    int* d_A;
    int* d_B;
    checkError(cudaMalloc(&d_A, nBytes));
    checkError(cudaMalloc(&d_B, 1024 * sizeof(int)));


    //warmup
    checkError( cudaMemcpy(d_A, &A[0], nBytes, cudaMemcpyHostToDevice) );
    checkError( cudaDeviceSynchronize() );
    reduce_neighbored_conflict_divergent <<< 1024, 512 >>> (d_A, d_B, nElem);
    checkError( cudaPeekAtLastError() );
    checkError( cudaDeviceSynchronize() );

    checkError( cudaMemcpy(d_A, &A[0], nBytes, cudaMemcpyHostToDevice) );
    checkError( cudaDeviceSynchronize() );
    reduce_neighbored_global <<< 1024, 512 >>> (d_A, d_B, nElem);
    checkError( cudaPeekAtLastError() );
    checkError( cudaDeviceSynchronize() );

    checkError( cudaMemcpy(d_A, &A[0], nBytes, cudaMemcpyHostToDevice) );
    checkError( cudaDeviceSynchronize() );
    reduce_neighbored_conflict_divergent <<< 1024, 512 >>> (d_A, d_B, nElem);
    checkError( cudaPeekAtLastError() );
    checkError( cudaDeviceSynchronize() );

    checkError( cudaMemcpy(d_A, &A[0], nBytes, cudaMemcpyHostToDevice) );
    checkError( cudaDeviceSynchronize() );
    reduce_neighbored_conflict_nondivergent <<< 1024, 512 >>> (d_A, d_B, nElem);
    checkError( cudaPeekAtLastError() );
    checkError( cudaDeviceSynchronize() );

    checkError( cudaMemcpy(d_A, &A[0], nBytes, cudaMemcpyHostToDevice) );
    checkError( cudaDeviceSynchronize() );
    reduce_interleaved_noconflict_nondivergent <<< 1024, 512 >>> (d_A, d_B, nElem);
    checkError( cudaPeekAtLastError() );
    checkError( cudaDeviceSynchronize() );

    checkError( cudaMemcpy(d_A, &A[0], nBytes, cudaMemcpyHostToDevice) );
    checkError( cudaDeviceSynchronize() );
    reduce_interleaved_noconflict_nondivergent_unrolled <<< 1024, 512 >>> (d_A, d_B, nElem);
    checkError( cudaPeekAtLastError() );
    checkError( cudaDeviceSynchronize() );

    checkError( cudaMemcpy(d_A, &A[0], nBytes, cudaMemcpyHostToDevice) );
    checkError( cudaDeviceSynchronize() );
    reduce_interleaved_noconflict_nondivergent_completelyunrolled<512> <<< 1024, 512 >>> (d_A, d_B, nElem);
    checkError( cudaPeekAtLastError() );
    checkError( cudaDeviceSynchronize() );

    checkError(cudaMemcpy(&B[0], d_B, 1024 * sizeof(int), cudaMemcpyDeviceToHost));

    for (long long i = 0; i < 1024; ++i)
    {
        if (B.at(i) != 2048)
        {
            std::cout << "error: " << i << " "
                      << B.at(i) << std::endl;
            exit(-1);
        }
    }

    checkError(cudaFree(d_A));
    checkError(cudaFree(d_B));
}
