#include <stdio.h>

#define CUDA_CHECK(cmd) do {                                      \
  cudaError_t err = (cmd);                                        \
  if (err != cudaSuccess) {                                         \
    fprintf(stderr, "CUDA ERROR %s at %s:%d: %s\n",               \
            #cmd, __FILE__, __LINE__, cudaGetErrorString(err));   \
      std::exit(EXIT_FAILURE);                                    \
  }                                                               \
} while (0)


__global__ void MatmulKernel(float* M, float* N, float* P, int width)
  // One output pixel is calculated per thread, so for a 4x4, that'll be 16 threads
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if ((row < width) && (col < width))
  {
    float p_value = 0;
    for (int k = 0; k < width; ++k)
    {
      p_value += M[row * width+k] * N[k * width + col];
    }
    P[row * width + col] = p_value;
  }
}

__global__ void MatmulKernelRowByRow(float* M, float* N, float* P, int width)
  // This kernel calculates one row of the output matrix per thread.
  // So for a 4x4 output matrix, there will be launched a total of 4 threads
{
  int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < width)
  {
    for (int k = 0; k < width; k++)
    {
      float p_value = 0.0f;
      for (int j = 0; j < width; j++)
      {
        p_value += M[row * width + j] * N[j * width + k];
      }

      P[row * width + k] = p_value;
    }
  }
}

__global__ void MatmulKernelColByCol(float* M, float* N, float* P, int width)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (col < width) 
  {
    for (int row = 0; row < width; ++row) 
    {
      float p_value = 0.0f;
      for (int k = 0; k < width; ++k) 
      {
        p_value += M[row * width + k] * N[k * width + col];
      }
      P[row * width + col] = p_value;
    }
  }
}

void matmul_rbr(float* M, float* N, float* P, int width)
{
  float *M_d = nullptr, *N_d = nullptr, *P_d = nullptr;
  size_t size = static_cast<size_t>(width) * width * sizeof(float);

  CUDA_CHECK(cudaMalloc((void**)&M_d, size));
  CUDA_CHECK(cudaMalloc((void**)&N_d, size));
  CUDA_CHECK(cudaMalloc((void**)&P_d, size));

  CUDA_CHECK(cudaMemcpy(M_d, M, size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(N_d, N, size, cudaMemcpyHostToDevice));

  MatmulKernelRowByRow<<<1, width>>>(M_d, N_d, P_d, width);

  CUDA_CHECK(cudaMemcpy(P, P_d, size, cudaMemcpyDeviceToHost));

  cudaFree(M_d);
  cudaFree(N_d);
  cudaFree(P_d);
}

void matmul_cbc(float* M, float* N, float* P, int width)
{
  float *M_d = nullptr, *N_d = nullptr, *P_d = nullptr;
  size_t size = static_cast<size_t>(width) * width * sizeof(float);

  CUDA_CHECK(cudaMalloc((void**)&M_d, size));
  CUDA_CHECK(cudaMalloc((void**)&N_d, size));
  CUDA_CHECK(cudaMalloc((void**)&P_d, size));

  CUDA_CHECK(cudaMemcpy(M_d, M, size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(N_d, N, size, cudaMemcpyHostToDevice));

  MatmulKernelColByCol<<<1, width>>>(M_d, N_d, P_d, width);

  CUDA_CHECK(cudaMemcpy(P, P_d, size, cudaMemcpyDeviceToHost));

  cudaFree(M_d);
  cudaFree(N_d);
  cudaFree(P_d);
}

int main(int argc, char** argv)
{
  float M[4][4] = {
    {2, 3, 2, 1},
    {4, 3, 5, 1},
    {1, 2, 2, 1},
    {4, 3, 2, 1}
  };

  float N[4][4] = {
    {2, 3, 2, 1},
    {4, 3, 2, 1},
    {1, 2, 2, 1},
    {4, 3, 2, 1}
  };

  float P[4][4] = {};
  float C[4][4] = {};

  matmul_rbr(&M[0][0], &N[0][0], &P[0][0], 4);
  matmul_cbc(&M[0][0], &N[0][0], &C[0][0], 4);

  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      printf("%6.1f ", P[r][c]);
    }
    printf("\n");
  }

  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      printf("%6.1f ", C[r][c]);
    }
    printf("\n");
  }
  return EXIT_SUCCESS;
}
