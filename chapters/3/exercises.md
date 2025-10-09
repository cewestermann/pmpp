# Exercises

1. In this chapter we implemented a matrix multiplication kernel that has each thread produce one output matrix element. In this question, you will implement different matrix-matrix multiplication kernels and compare them.

    c. Analyze the pros and cons of each of the two kernel designs (row by row vs col by col)
        
        Doing row by row or col by col will suboptimally result in fewer threads and lower computation speeds. Since for a 4x4 matrix, we'll only have 4 threads in both cases as compared to 16 threads for the 2D case. Furthermore, row by row is the worst outcome, since the column elements that you are accessing in the inner loop are not contiguous in memory, which they are in the col by col version.

2. Write a matrix-vector multiplication kernel and the host stub function that can be called with four parameters: pointer to the output matrix, pointer to the input matrix, pointer to the input vector, and the number of elements in each dimension. Use one thread to calculate an output vector element.

See matvecmul.cu

3. Consider the following CUDA kernel and the corresponding host function that calls it:

```c
    __global__ void foo_kernel(float* a, float* b, unsigned int M, unsigned int N)
    {
        unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
        unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row < M && col < N)
        {
            b[row*N + col] = a[row*N + col]/2.1f + 4.8f;
        }
    }

    void foo(float* a_d, float* b_d)
    {
        unsigned int M = 150;
        unsinged int N = 300;
        dim3 bd(16, 32);
        dim3 gd((N - 1) / 16 + 1, (M - 1) / 32 + 1);
        foo_kernel <<< gd, bd >>>(a_d, b_d, M, N);
    }
```

    Let's write out the calculations:

        dim3 bd(16, 32);
        dim3 gd((N - 1) / 16 + 1, (M - 1) / 32 + 1);

    is equal to

        Threads in each block:
            dim3 bd(16, 32);
        Blocks in the grid:
            dim3 gd(19.6875, 5.65625) -> gd(19, 5);

    a. What is the number of threads per block?
        16 * 32 = 512

    b. What is the number of threads in the grid?
        19 * 5 * 512 = 48.640

    c. What is the number of blocks in the grid?
        19 * 5 = 95

    d. What is the number of threads that execute the code on line 05?
        max(threadIdx.x) = 15
        max(blockDim.x) = 16
        max(blockIdx.x) = 18

        blockIdx.x * blockDim.x + threadIdx.x = 18 * 16 + 15 = 303

        max(threadIdx.y) = 31
        max(blockDim.y) = 4
        max(blockIdx.y) = 32
        
        blockIdx.y * blockDim.y + threadIdx.y = 32 * 4 + 31 = 159
        
        M = 150, N = 300. M x N = 45000 which is lower than 303 x 159 = 48640, so 45000 threads will run line 05
        
4. Consider a 2D matrix A with a width of 400, a height of 500. The matrix is stored as a one-dimensional array. Specify the array index of the matrix element at row 20 and column 10:
    a. If the matrix is stored in row-major order
        A[row * width + column] 
        A[20 * 400 + 10] = 8010
        
    b. If the matrix is stored in column-major order
        A[col * height + row]
        A[10 * 500 + 20] = 5020
        
5. Consider a 3D tensor A with a width of 400, a height of 500 and a depth of 300. The tensor is stored as a one-dimensional array in row-major order. Specify the array index of the tensor element at
    x = 10, y = 20, z = 5

    A[plane * width * height + x * width + y]
    A[5 * 400 * 500 + 20 * 400 + 10] = 1.008.010

    



