// Ignore path not found error
#include "crisprme2/include/kernels.hh"

#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>
namespace cg = cooperative_groups;

#define DEBUG 0

// Mining kernels
#include "mine.cu"

#include <stdio.h>
#include <stdint.h>

using u32 = uint32_t;
using s32 = int32_t;
using u8  = uint8_t;

#define cuda_check_error() {                                          \
    cudaError_t e = cudaGetLastError();                                 \
    if(e != cudaSuccess) {                                              \
        printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
        exit(0); \
    }                                                                 \
}

__global__ void mine0(unsigned char* data, int N) {
    auto block = cg::this_thread_block();
    if (block.thread_rank() == 0)
        printf("Kernel running!\n");
}

/*

#define QSTR "TTACCAGATTACCAGA"
#define QLEN 16

#define TSTR "GATTACAGATTACCAGATTACCAGATTACCA"
#define TLEN 32

#define QLEN_MAX 32
#define TLEN_MAX 32

#define GAPS 2
#define MISS 2

/// An alignment state
union state_t {
    u32 merged;
    struct {
        u8 qidx; // Index into the query string
        u8 tidx; // Index into the reference string
        u8 gaps; // Number of gaps
        u8 miss; // Number of missmatches
    } parts;
};

__device__ __forceinline__ void print_state(s32 lane, state_t state) {
    printf("thread %2d, state: { qidx: %2d, tidx: %2d, gaps: %2d, miss: %2d, }\n",
        lane, state.parts.qidx, state.parts.tidx, state.parts.gaps, state.parts.miss);
}

__device__ __forceinline__ bool state_is_valid(state_t state, int qlen, int tlen) {
    return state.parts.qidx <  qlen
        && state.parts.tidx <  tlen
        && state.parts.gaps <= GAPS
        && state.parts.miss <= MISS;
}

__device__ __forceinline__ bool state_is_final(state_t state, int qlen) {
    return state.parts.qidx == qlen - 1;
}

template<u32 WARP_COUNT, u32 FRONTIER_SIZE>
__global__ void filter1(
    unsigned char* __restrict global_query,
    unsigned char* __restrict global_strings,
    int qlen,
    int tlen,
    int n
) {

    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile warp = cg::tiled_partition<32>(block);
    s32 warp_id = warp.meta_group_rank();
    s32 lane = warp.thread_rank();

    // Query string to search
    __shared__ u8 query[QLEN_MAX];
    if (warp_id == 0 && lane < qlen)
        query[lane] = global_query[lane];

    block.sync();

    // Target string for each warp
    __shared__ u8 target[WARP_COUNT][TLEN_MAX];
    if (lane < tlen) 
        target[warp_id][lane] = global_strings[(blockIdx.x * blockDim.x) + (warp_id * tlen) + lane];

    block.sync();

    // Exploration frontier
    s32 frontier_size = 1;
    __shared__ state_t frontier[WARP_COUNT][FRONTIER_SIZE];
    if (warp.thread_rank() == 0)
        frontier[warp_id][0] = state_t { 0 };

    // Bloom filter (big as the frontier)
    __shared__ u32 bloom[WARP_COUNT][FRONTIER_SIZE];

    state_t next_state;
    bool next_state_valid = false;
    bool next_state_final = false;
    bool next_state_bloom = true;

    u32 expansion = 1;
    while(frontier_size != 0) {
        warp.sync();

        state_t current_state = state_t { 0xDEADFEED };

        // =========================================================
        // FRONTIER POP

        // Extract an item from the frontier
        u32 frontier_popped = 0;
        if (frontier_size - lane - 1 >= 0) {
            current_state = frontier[warp_id][frontier_size - lane - 1];
            frontier_popped = 1;
        }

        printf("[exp. %4d][warp_id %2d][lane %2d][extract]: (%2d,%2d,%2d,%2d) (%x)\n",
            expansion, warp_id, lane, 
            current_state.parts.qidx, 
            current_state.parts.tidx, 
            current_state.parts.gaps, 
            current_state.parts.miss,
            current_state.merged
        );

#if DEBUG
        printf("exp. %2d(pop), thread %2d, frontier_popped = %d\n", 
            expansion, lane, frontier_popped);
        warp.sync();
#endif

        // Calculate how many elements have been removed
        #pragma unroll(5)
        for (auto i = 1; i <= 16; i *= 2) {
            frontier_popped += warp.shfl_xor(frontier_popped, i);
        }
        frontier_size -= frontier_popped;

#if DEBUG
        printf("exp. %2d(pop), thread %2d, frontier_popped_acc = %d\n", 
            expansion, lane, frontier_popped);
        warp.sync();
#endif

        // =========================================================
        // MATCH/MISMATCH NODES

        // Generate match/mismatch nodes
        if (current_state.merged != 0xDEADFEED) {
            
            next_state = current_state;
            next_state.parts.qidx += 1;
            next_state.parts.tidx += 1;

            // Increase mismatch score if query and target string are not equal
            auto miss = (query[current_state.parts.qidx] != target[warp_id][current_state.parts.tidx]) ? 1 : 0;
            next_state.parts.miss += miss;

            // Check if the next state is valid, is a solution
            next_state_valid = state_is_valid(next_state, qlen, tlen);
            next_state_final = state_is_final(next_state, qlen);
        }

#if DEBUG
        printf("exp. %2d(match-mismatch), thread %2d, valid = %d, final = %d\n", 
            expansion, lane, next_state_valid, next_state_final);
        warp.sync();
#endif

        // Check if any thread in the warp has a solution
        bool next_state_solution = next_state_valid && next_state_final;
        if (warp.any(next_state_solution)) {
            if (next_state_solution)
                print_state(lane, next_state);

            // Entire warp exits
            return;
        }

        // Add match/mismatch nodes to the frontier
        if (current_state.merged != 0xDEADFEED) { 
            if (next_state_valid) {
                // Threads that have a valid next state
                auto g = cg::coalesced_threads();

#if DEBUG
                printf("exp. %2d(match-mismatch), thread %2d, write_idx = %d\n", 
                    expansion, lane, frontier_size + g.thread_rank());
#endif

                frontier[warp_id][frontier_size + g.thread_rank()] = next_state;
                printf("[exp. %4d][warp_id %2d][lane %2d][insert %-10s]: (%2d,%2d,%2d,%2d)\n",
                    expansion, warp_id, lane, "match",
                    next_state.parts.qidx, 
                    next_state.parts.tidx, 
                    next_state.parts.gaps, 
                    next_state.parts.miss
                );
            }
        }

#if DEBUG
        warp.sync();
#endif

        // Update current frontier write index
        u32 total_next_states = (u32)next_state_valid;
        #pragma unroll(5)
        for (auto i = 1; i <= 16; i *= 2) {
            total_next_states += warp.shfl_xor(total_next_states, i);
        }
        frontier_size += total_next_states;

#if DEBUG
        printf("exp. %2d(match-mismatch), thread %2d, total_next_states = %d\n", 
            expansion, lane, total_next_states);
        warp.sync();
#endif

        // =========================================================
        // TARGET GAP NODES

        // Generate target gap nodes
        if (current_state.merged != 0xDEADFEED) {
            
            next_state = current_state;
            next_state.parts.tidx += 1;

            // Increase gap score if we are not on the first element of the query 
            auto gaps = (current_state.parts.qidx != 0) ? 1 : 0;
            next_state.parts.gaps += gaps;

            // Check if the next state is valid and is a solution
            next_state_valid = state_is_valid(next_state, qlen, tlen);
            next_state_final = state_is_final(next_state, qlen);
        }

#if DEBUG
        printf("exp. %2d(target-gap), thread %2d, valid = %d, final = %d\n", 
            expansion, lane, next_state_valid, next_state_final);
        warp.sync();
#endif

        // Check if any thread in the warp has a solution
        next_state_solution = next_state_valid && next_state_final;
        if (warp.any(next_state_solution)) {
            if (next_state_solution)
                print_state(lane, next_state);

            // Entire warp exits
            return;
        }

        // Add target gap nodes to the frontier
        if (current_state.merged != 0xDEADFEED) { 
            if (next_state_valid) {
                // Threads that have a valid next state
                auto g = cg::coalesced_threads();

#if DEBUG
                printf("exp. %2d(target-gap), thread %2d, write_idx = %d\n", 
                    expansion, lane, frontier_size + g.thread_rank());
#endif

                frontier[warp_id][frontier_size + g.thread_rank()] = next_state;
                printf("[exp. %4d][warp_id %2d][lane %2d][insert %-10s]: (%2d,%2d,%2d,%2d)\n",
                    expansion, warp_id, lane, "target-gap",
                    next_state.parts.qidx, 
                    next_state.parts.tidx, 
                    next_state.parts.gaps, 
                    next_state.parts.miss
                );
            }
        }

#if DEBUG
        warp.sync();
#endif

        // Update current frontier write index
        total_next_states = (u32)next_state_valid;
        #pragma unroll(5)
        for (auto i = 1; i <= 16; i *= 2) {
            total_next_states += warp.shfl_xor(total_next_states, i);
        }
        frontier_size += total_next_states;

#if DEBUG
        printf("exp. %2d(target-gap), thread %2d, total_next_states = %d\n", 
            expansion, lane, total_next_states);
        warp.sync();
#endif

        // =========================================================
        // QUERY GAP NODES

        // Generate query gap nodes
        if (current_state.merged != 0xDEADFEED) {
            
            next_state = current_state;
            next_state.parts.qidx += 1;
            next_state.parts.miss += 1;

            // Check if the next state is valid and is a solution
            next_state_valid = state_is_valid(next_state, qlen, tlen);
            next_state_final = state_is_final(next_state, qlen);
        }

#if DEBUG
        printf("exp. %2d(query-gap), thread %2d, valid = %d, final = %d\n", 
            expansion, lane, next_state_valid, next_state_final);
        warp.sync();
#endif

        // Check if any thread in the warp has a solution
        next_state_solution = next_state_valid && next_state_final;
        if (warp.any(next_state_solution)) {
            if (next_state_solution)
                print_state(lane, next_state);

            // Entire warp exits
            return;
        }

        // Add query gap nodes to the frontier
        if (current_state.merged != 0xDEADFEED) { 
            if (next_state_valid) {
                // Threads that have a valid next state
                auto g = cg::coalesced_threads();

#if DEBUG
                printf("exp. %2d(query-gap), thread %2d, write_idx = %d\n", 
                    expansion, lane, frontier_size + g.thread_rank());
#endif

                frontier[warp_id][frontier_size + g.thread_rank()] = next_state;
                printf("[exp. %4d][warp_id %2d][lane %2d][insert %-10s]: (%2d,%2d,%2d,%2d)\n",
                    expansion, warp_id, lane, "query-gap",
                    next_state.parts.qidx, 
                    next_state.parts.tidx, 
                    next_state.parts.gaps, 
                    next_state.parts.miss
                );
            }
        }

#if DEBUG
        warp.sync();
#endif

        // Update current frontier write index
        total_next_states = (u32)next_state_valid;
        #pragma unroll(5)
        for (auto i = 1; i <= 16; i *= 2) {
            total_next_states += warp.shfl_xor(total_next_states, i);
        }
        frontier_size += total_next_states;

#if DEBUG
        printf("exp. %2d(query-gap), thread %2d, total_next_states = %d\n", 
            expansion, lane, total_next_states);
        warp.sync();
#endif

        // =========================================================
        expansion += 1;

        if (expansion > 4000 && lane == 0) {
            if (lane == 0)
                printf("WARNING: HIGH EXPANSIONS! (%d)\n", expansion);
            return;
        }

        if (warp.thread_rank() == 0) {
            printf("==========================================\n");
        }
    }
    warp.sync();
}



void mine(unsigned char* data, int N) {

    cudaDeviceSynchronize();
}

/*
void filter(const unsigned char* query, const unsigned char* trgts, int qlen, int tlen, int n) {

    unsigned char *dev_query, *dev_trgts;

    cudaMalloc(&dev_query, qlen);
    cudaMalloc(&dev_trgts, tlen * n);

    // Copy CPU memory to GPU
    cudaMemcpy(dev_query, query, sizeof(u8) * qlen,     cudaMemcpyHostToDevice);
    cudaMemcpy(dev_trgts, trgts, sizeof(u8) * tlen * n, cudaMemcpyHostToDevice);

    constexpr u32 warp_size = 32;
    constexpr u32 warp_count = 1;

    u32 blocks = (n + warp_count - 1) / warp_count;
    printf("launching kernel with %d blocks\n", blocks);

    filter0<warp_count, 256><<<blocks, warp_count * warp_size>>>(
        dev_query,
        dev_trgts,
        qlen,
        tlen,
        n
    );

    cuda_check_error();
    cudaDeviceSynchronize();
}
*/

void filter(const u8* query, const u8* strings, u8* result, int qlen, int tlen, int n) {
    unsigned char *dev_query, *dev_strings, *dev_result;

    cudaMalloc(&dev_result, n);
    cudaMalloc(&dev_strings, tlen * n);
    cudaMalloc(&dev_query, qlen);

    // Copy CPU memory to GPU
    cudaMemcpy(dev_strings, strings, sizeof(u8) * tlen * n, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_query, query, sizeof(u8) * qlen, cudaMemcpyHostToDevice);

    constexpr u32 warp_size = 32;
    constexpr u32 warp_count = 2;

    int max_active_blocks = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks, filter2<warp_count, 24, 24>, warp_count * warp_size, 0);
    printf("max active blocks: %d\n", max_active_blocks);

    u32 blocks = (n + warp_count * warp_size - 1) / (warp_count * warp_size);
    printf("launching kernel with %d blocks\n", blocks);

    cudaEvent_t start, stop; 
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

#define LAPS_COUNT 100

    float laps_acc = 0.0f;
    for (u32 lap = 0; lap < LAPS_COUNT; ++lap) {

        cudaEventRecord(start, 0);
        filter1<warp_count, 24, 24><<<blocks, warp_count * warp_size>>>(
            dev_query,
            dev_strings,
            dev_result,
            n
        );

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        float time;
        cudaEventElapsedTime(&time, start, stop);
        laps_acc += time;
    }

    printf("elapsed kernel time (%d reps): %.2f ms\n", LAPS_COUNT, laps_acc / LAPS_COUNT);


    // Get result for GPU
    cudaMemcpy(result, dev_result, n, cudaMemcpyDeviceToHost);
    cuda_check_error();

    cudaFree(dev_query);
    cudaFree(dev_strings);
    cudaFree(dev_result);
}

/*
void filter_fixed(const u8* query, const u8* strings, u8* result, int qlen, int tlen, int n) {
    
    constexpr u32 blocks_mult = 32; 
    constexpr u32 warp_size   = 32;
    constexpr u32 warp_count  = 2;
    
    u32 sm_count;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, devId);
    u32 blocks = sm_count * blocks_mult;

    unsigned char *dev_query, *dev_strings, *dev_result;

    cudaMalloc(&dev_result, n);
    cudaMalloc(&dev_strings, tlen * n);
    cudaMalloc(&dev_query, qlen);

    // Copy CPU memory to GPU
    cudaMemcpy(dev_strings, strings, sizeof(u8) * tlen * n, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_query, query, sizeof(u8) * qlen, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop; 
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    filter2<warp_count, 24, 24><<<blocks, warp_count * warp_size>>>(
        dev_query,
        dev_strings,
        dev_result,
        n
    );
    cudaEventRecord(stop, 0);

    float time;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("kernel elapsed time: %3.1f ms \n", time);

    // Get result for GPU
    cudaMemcpy(result, dev_result, n, cudaMemcpyDeviceToHost);
    cuda_check_error();

    cudaFree(dev_query);
    cudaFree(dev_strings);
    cudaFree(dev_result);
}
*/