#include "kernels.h"
#include <stdio.h>

__global__ void testKernel() {
    printf("Hello from GPU Thread %d!\n", threadIdx.x);
}

void launchTestKernel() {
    // Launch 1 block with 5 threads
    testKernel<<<1, 5>>>();
    cudaDeviceSynchronize();
}