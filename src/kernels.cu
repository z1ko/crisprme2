// Ignore path not found error
#include "crisprme2/include/kernels.hh"

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#include <stdio.h>

__global__ void mine0(unsigned char* data, int N) {
    auto block = cg::this_thread_block();
    if (block.thread_rank() == 0)
        printf("Kernel running!\n");
}


void mine(unsigned char* data, int N) {

    mine0<<<1, 1024>>>(data, N);
    mine0<<<1, 1024>>>(data, N);
    mine0<<<1, 1024>>>(data, N);
    mine0<<<1, 1024>>>(data, N);
    
    cudaDeviceSynchronize();
}