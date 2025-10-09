#include <stdio.h>

#define CUDA_CHECK(cmd) do {                                      \
  cudaError_t err = (cmd);                                        \
  if (err != cudaSuccess) {                                       \
    fprintf(stderr, "CUDA ERROR %s at %s:%d: %s\n",               \
            #cmd, __FILE__, __LINE__, cudaGetErrorString(err));   \
      std::exit(EXIT_FAILURE);                                    \
  }                                                               \
} while (0)


__global__ void MatVecMul(float* M, float* v, float* v_out, int n)
{
  // What output element is the thread calculating (And thus what row of M we are using)
  int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < n)
  {
    float sum = 0.0f;
    for (int k = 0; k < n; ++k)
    {
      sum += M[row * n + k] * v[k];
    }
    v_out[row] = sum;
  }
}

void matvec(float* M_h, float* v_h, float* v_out, int n)
{
  float* M_d = nullptr;
  float* v_d = nullptr;
  float* v_out_d = nullptr;

  // We assume only square matrices
  size_t size_mat = size_t(n) * n * sizeof(float);
  size_t size_vec = size_t(n) * sizeof(float);

  CUDA_CHECK(cudaMalloc((void**)&M_d, size_mat));
  CUDA_CHECK(cudaMalloc((void**)&v_d, size_vec));
  CUDA_CHECK(cudaMalloc((void**)&v_out_d, size_vec));

  CUDA_CHECK(cudaMemcpy(M_d, M_h, size_mat, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(v_d, v_h, size_vec, cudaMemcpyHostToDevice));

  MatVecMul<<<1, n>>>(M_d, v_d, v_out_d, n);

  CUDA_CHECK(cudaMemcpy(v_out, v_out_d, size_vec, cudaMemcpyDeviceToHost));

  cudaFree(M_d);
  cudaFree(v_d);
  cudaFree(v_out_d);
}

int main(int argc, char** argv)
{
  float M[3][3] = {
    {1, 2, 3},
    {2, 3, 4},
    {3, 4, 5}
  };

  float v[3] = {5, 8, 10};

  float v_out[3] = {};

  matvec(&M[0][0], v, v_out, 3);

  for (int r = 0; r < 3; ++r)
  {
    printf("%6.1f ", v_out[r]);
  }

  return EXIT_SUCCESS;
}
