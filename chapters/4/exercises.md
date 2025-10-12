# Exercises for Compute Architecture and Scheduling

1. Consider the following CUDA kernel and the corresponding host function that calls it:

```c
__global__ void foo_kernel(int* a, int*b) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.x < 40 || threadIdx.x >= 104) {
        b[i] = a[i] + 1;
    }

    if (i%2 == 0) {
        a[i] = b[i] * 2;
    }

    for (unsigned int j = 0; j < 5 - (i % 3); ++j) {
        b[i] += j;    
    }
}

void foo(int* a_d, int* b_d) {
    unsigned int N = 1024;
    foo_kernel<<< (N + 128 - 1) / 128, 128 >>>(a_d, b_d);
}
```

    a. What is the number of warps per block?
        We assume that a warp is considered to be 32 threads as that is what most implementations do.
        We have 128 threads per block, which means that we must have 128 / 32 = 4 warps per block

    b. What is the number of warps in the grid?
        We are just short of 9 blocks, and since we don't round up, we actually have 8 blocks
        8 blocks of 128 threads = 1024
        1024 / 32 = 32 warps in the grid

    c. For the statement on line 04:
        i. How many warps in the grid are active?

            We are talking about the body of this conditional.
            We have 128 threads in a block, so the threadIdx will range from [0, 127].

            if (threadIdx.x < 40 || threadIdx.x >= 104) {
                b[i] = a[i] + 1;
            }

            That means, that threads 0 - 39 inclusive and 104 - 127 inclusive are active for the
            body of the conditional.

            Since warps are 32 threads and they are consecutive (I assume)? We will have two warps
            that satisfy the first part of the or conditional (warp 0: 0 - 31, warp 1: 32 - 63) but only parts of warp 1 will execute, 
            and we will have 1 warp that execute the last part of the or conditional, warp 3 (96 - 127 inclusive). Warp 2
            is in the middle of the conditional, so it will not be assigned to execute the body.

        ii. How many warps in the grid are divergent?

            Divergence means that when we condition a statement on threadIdx, we create
            separate code paths for different warps. This will cause multiple passes, as 
            we will have one pass where the warps execute when condition is truthy, and then
            another pass where the warps execute when the condition is false.

            We know we have 32 warps in the grid and only 3 warps will activate for line 04.
            So we have 3 divergent warps.

        iii. What is the SIMD efficiency (in %) of warp 0 of block 0?
            Warp 0 of block 0 will have 100% efficiency, since all threads in the warp
            will execute the statement (threads 0 - 31 inclusive)

        iv. What is the SIMD efficiency (in %) of warp 1 of block 0?
            Warp 1 will have 8 threads executing out of 32, i.e, 8 / 32 = 25% SIMD efficiency.

        v. What is the SIMD efficiency (in %) of warp 3 of block 0?
            Warp 3 will have 128 - 104 = 24 threads out of 32 execute, -> 24 / 32 = 75% SIMD efficiency.

    d. For the statement on line 07:

            if (i%2 == 0) {
                a[i] = b[i] * 2;
            }

        where: unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

        i. How many warps in the grid are active?

        Every second thread will execute, but that means that all warps will work on this statement

        ii. How many warps in the grid are divergent?
        All are divergent.

        iii. What is the SIMD efficiency (in %) of warp 0 of block 0?

        50% SIMD efficiency, since only every second thread in the warp will execute the statement

    e. For the loop on line 09:

            for (unsigned int j = 0; j < 5 - (i % 3); ++j) {
                b[i] += j;    
            }

        i. How many iterations have no divergence?

            2/3 of the iterations have no divergence
        ii. How many iterations have divergence?
            1/3 of the iterations have divergence


2. For a vector addition, assume that the vector length is 2000, each thread calculates one output
element, and the thread block size is 512 threads. How many threads will be in the grid?

2048 threads will be in the grid.

3. For the previous question, how many warps do you expect to have divergence due to the boundary check on vector length?
I only expect one warp to have divergence, specifically warp 62, which is thread 1984 - 2016. The final warp will not
be active at all, so no divergence either

4. Consider a hypothetical block with 8 threads executing a section of code before reaching a barrier.
The threads require the following amount of time (in microseconds) to execute the sections: 
2.0, 2.3, 3.0, 2.8, 2.4, 1.9, 2.6, and 2.9; they spend the rest of their time waiting for the barrier.
What percentage of the threads' total execution time is spent waiting for the barrer?

The longest execution time is 3.0 microseconds, which means that the total time spent waiting is:

3.0 - 2.0 + 3.0 - 2.3 + 3.0 - 2.8 + 3.0 - 2.4 + 3.0 - 1.9 + 3.0 - 2.6 + 3.0 - 2.9 = 4.1 microseconds

Out of a total of

8x3 microseconds, so 4.1 / 24 = 16.6% time is spent waiting

5. A CUDA programmer says that if they launch a kernel with only 32 threads in each block, they can
leave out the syncthreads() instruction wherever barrier synchronization is needed. Do you think this is a good
idea? Explain.

It sounds reasonable, since each block will also be a warp and thus all threads within the warp will be 
executed in lockstep (at the same time.). At the same time, I don't think it is guaranteed that 
they will finish their jobs in the exact same instant (maybe do to memory reads/writes?).

Furthermore, if NVIDIA decides to change warp size to something else than 32 threads, having a
syncthreads() intrinsic in there might future-proof the code better.

6. If a CUDA device's SM can take up to 1536 threads and up to 4 thread blocks, which of the following
block configurations would result in the most number of threads in the SM?

    a. 128 threads per block
    b. 256 threads per block
    c. 512 threads per block
    d. 1024 threads per block


I'd assume c, 512 threads per block. We have one block less, but will utilize all 1536 threads.

7. Assume a device that allows up to 64 blocks per SM and 2048 threads per SM.
Indicate which of the following assignments per SM are possible. In the cases in which it is 
possible, indicate the occupancy level.

    a. 8 blocks with 128 threads each,  8x128 = 1024, 50%
    b. 16 blocks with 64 threads each,  16x64 = 1024, 50%
    c. 32 blocks with 32 threads each,  32x32 = 1024, 50%
    d. 64 blocks with 32 threads each,  64*32 = 2048, 100%
    e. 32 blocks with 64 threads each,  32*64 = 2048, 100%

8. Consider a GPU with the following hardware limits: 

2048 threads per SM
32 blocks per SM
64K (65536) registers per SM.

For each of the following kernel characteristics, specify whether the kernel can achieve full
occupancy. If not, specify the limiting factor.

    a. The kernel uses 128 threads per block and 30 registers per thread.

        2048/128 = 16 blocks -> Valid
        2048 * 30 = 61440 -> Valid

    b. The kernel uses 32 threads per block and 29 registers per thread.
        
        2048/32 = 64 blocks -> Not valid, the SM only allows 32 blocks
        2048 * 29 = 59392 -> Valid

    c. The kernel uses 256 threads per block and 34 registers per thread.

        2048 / 256 = 8 blocks -> Valid
        2048 * 34 = 69632 -> Not valid. The SM only allows 65536 registers (2^16)

9. A student mentions that they were able to multiply two 1024x1024 matrices using a matrix multiplication
kernel with 32x32 thread blocks. The student is using a CUDA device that allows up to 512 threads per block
and up to 8 blocks per SM. The student further mentions that each thread in a thread block calculates
one element of the result matrix. What would be your reaction and why?

A grid with 32x32 (1024 blocks) of 512 threads each = 512 * 1024 = 524,288.
There are 1024 x 1024 = 1,048,576 elements in the result matrix which means there are not
enough total threads to calculate an element each.


















