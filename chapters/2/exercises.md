1. If we want to use each thread in a grid to calculate one output element of a vector addition, what would be the expression for mapping the thread/block indices to the data index (i)?

    (A) ```i = threadIdx.x + threadIdx.y;```
    (B) ```i = blockIdx.x + threadIdx.x;```
    (C) ```i = blockIdx.x*blockDim.x + threadIdx.x```
    (D) ```i = blockIdx.x * threadIdx.y;

The answer is (C), as the data index is a global index for a given thread. Given that the grid consists of say, 3 blocks and 256 threads in each block, getting thread number 8 in block 2 would involve calculating ```i = 2 * 256 + 8``` which would be global thread ID 520

2. Assume that we want to use each thread to calculate two adjacent elements of a vector addition. What would be the expression for mapping the thread/block indices to the data index (i) of the first element to be processed by a thread?

    (A) ```i = blockIdx.x*blockDim.x + threadIdx.x + 2;```
    (B) ```i = blockIdx.x*threadIdx.x*2;```
    (C) ```i = (blockIdx.x*blockDim.x + threadIdx.x)*2;```
    (D) ```i = blockIdx.x*blockDim.x*2 + threadIdx.x;```


Let's set up a simple case: The grid will consist of 3 blocks and 8 threads in each block. Say we want the first thread that handles elements 0 and 1:

Using A: ```0 * 8 + 0 + 2 = 2```
         ```0 * 8 + 1 + 2 = 3```

A only increments elements by 1, we would expect increments of 2

Using B: ```0 * 0 * 2 = 0``` (This will always be zero for block 0, no good)

Using C: ```(0 * 8 + 0) * 2 = 0```
         ```(0 * 8 + 1) * 2 = 2```
         ```(0 * 8 + 2) * 2 = 4```

C seems to be the right one as each thread is assigned two elements as we can see above.
         
3. We want to use each thread to calculate two elements of a vector addition. Each thread block processes ```2 * blockDim.x``` consecutive elements that form two sections. All threads in each block will process a section first, each processing one element. They will then all move to the next section, each processing one element. Assume that variable ```i``` should be the index for the first element to be processed by a thread. What would be the expression for mapping the thread/block indices to data index of the first element?

    (A) ```i = blockIdx.x*blockDim.x + threadIdx.x + 2;```
    (B) ```i = blockIdx.x*threadIdx.x*2;```
    (C) ```i = (blockIdx.x*blockDim.x + threadIdx.x)*2;```
    (D) ```i = blockIdx.x*blockDim.x*2 + threadIdx.x;```

D is the correct answer. 

4. For a vector addition, assume that the vector length is 8000, each thread calculates one output element, and the thread block size is 1024 threads. The programmer configures the kernel call to have a minimum number of thread blocks to cover all output elements. How many threads will be in the grid?

    (A) 8000
    (B) 8196
    (C) 8192
    (D) 8200

We need at least 8 thread blocks of size 1024 to call all 8000 elements. That'll mean 8 * 1024 = 8192, so (C). We will shut off the remaining threads though, using a guard statement such as

```if (i > n) return;```

5. If we want to allocate an array of v integer elements in the CUDA device global memory, what would be an appropriate expression for the second argument of the cudaMalloc call?

    (A) n
    (B) v
    (C) n * sizeof(int)
    (D) v * sizeof(int)

The answer is (D), ```v * sizeof(int)```

6. If we want to allocate an array of n floating-point elements and have a floating-point pointer variable A_d to point to the allocated memory, what would be an appropriate expression for the first argument of the cudaMalloc() call?

    (A) n
    (B) (void*)A_d
    (C) *A_d
    (D) (void**)&A_d

We would need a pointer to the pointer, i.e., (D) ```(void**)&A_d``` such that cudaMalloc can take our pointer and point it to the allocated device memory.

7. If we want to copy 3000 bytes of data from host array A_h (A_h is a pointer to element 0 of the source array) to device array A_d (A_d is a pointer to element 0 of the destination array), what would be an appropriate API call for this data copy in CUDA?

    (A) ```cudaMemcpy(3000, A_h, A_d, cudaMemcpyHostToDevice)```
    (B) ```cudaMemcpy(A_h, A_d, 3000, cudaMemcpyDeviceToHost)```
    (C) ```cudaMemcpy(A_d, A_h, 3000, cudaMemcpyHostToDevice)```
    (D) ```cudaMemcpy(3000, A_d, A_h, cudaMemcpyHostToDevice)```

The cudaMemcpy definition is (dst, src, count, transfer_type), so, we want (A_d, A_h, 3000, cudaMemcpyHostToDevice), i.e., (C).

8. How would one declare a variable ```err``` that can appropriately receive the returned value of a CUDA API call?
    
    (A) ```int err```
    (B) ```cudaError err```
    (C) ```cudaError_t err```
    (D) ```cudaSuccess_t err```

Obviously not (D)... The correct type is ```cudaError_t```, so (C).

9. Consider the following CUDA kernel and the corresponding host function that calls it:

```c
__global__ void foo_kernel(float* a, float* b, unsigned int N) 
{
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    if (i < N) {
        b[i] = 2.7f * a[i] - 4.3f;
    }
}

void foo(float* a_d, float* b_d)
{
    unsigned int N=200000;
    foo_kernel<<<(N + 128 - 1)/128, 128>>>(a_d, b_d, N);
}
```

    a. What is the number of threads per block?
        128 (from <<<_, 128>>>)

    b. What is the number of threads in the grid?
        200000

    c. What is the number of blocks in the grid?
        (200000 + 128 - 1) / 128 = 1563.4922. We don't round up, so I assume 1563 blocks
    d. What is the number of threads that execute the code on line 02?
        200064
    e. What is the number of threads that execute the code on line 04?
        200000

10. A new summer intern was frustrated with CUDA. He has been complaining that CUDA is very tedious. He had to declare many functions that he plans to execute on both the host and the device twice, once as a host function and once as a device function. What is your response?


That there is no need to have several of the same function, as you can specify both ```__host__``` and ```__device__``` on the same function, leading to only one definition per function.
