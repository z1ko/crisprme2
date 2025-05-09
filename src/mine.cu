#include <stdint.h>
#include <stdio.h>

#define DEBUG_STRINGS 0
#define DEBUG_QUERY 0

using u32 = uint32_t;
using s32 = int32_t;
using u8  = uint8_t;

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

constexpr u32 warp_size = 32;

constexpr u32 cost_gaps = 1;
constexpr u32 cost_miss = 1;

/// Simplest functional kernels, this uses shared memory for both strings, query and tmp results
template<u32 warp_count, u32 qlen, u32 tlen>
__global__ void filter0(u8 *gquery, u8 *gstrings, u8* result, u32 N) {

    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile warp = cg::tiled_partition<warp_size>(block);
    s32 warp_id = warp.meta_group_rank();
    s32 lane = warp.thread_rank();

    // Keep strings and query in shared memory for fast access
    __shared__ u8 strings[warp_count][warp_size][tlen];
    __shared__ u8 query[qlen];

    // Last computed page in the DPT table
    __shared__ u8 cache[warp_count][2][qlen][warp_size];
    u32 cache_curr_page = 0;

    // Load strings, each warp load its strings
    #pragma unroll(warp_size)
    for (u32 sidx = 0; sidx < warp_size; ++sidx)
        if (lane < tlen) {
            const auto global_idx = (blockDim.x * blockIdx.x * tlen) + (warp_id * warp_size * tlen) + (sidx * tlen); 
            strings[warp_id][sidx][lane] = gstrings[global_idx + lane];
        }

    // Load query for all the warps
    if (warp_id == 0 && lane < qlen)
        query[lane] = gquery[lane];

    // Initialize cache with base values
    #pragma unroll(qlen)
    for (u32 qidx = 0; qidx < qlen; ++qidx)
        cache[warp_id][cache_curr_page][qidx][lane] = qidx + 1;

#if DEBUG
    block.sync();
    if (warp_id == 0 && lane == 0) {
        
        // Print strings
        printf("strings:\n");
        for (u32 i = 0; i < warp_size; ++i) {
            printf(" %2d ", i);
            for (u32 s = 0; s < tlen; ++s) {
                printf("%c", strings[warp_id][i][s]);
            }
            printf("\n");
        }

        // Print query
        printf("query:\n    ");
        for (u32 s = 0; s < qlen; ++s) {
            printf("%c", query[s]);
        }
        printf("\n");
    }
#endif

    // All warps must have completed the above steps
    block.sync();

    // ==============================================================
    // Alignment

    u8 best_global_score = 255;

    #pragma unroll(tlen)
    for (u32 tidx = 0; tidx < tlen; ++tidx) {

        // Advance cache circular buffer
        // NOTE: underflow is OK
        u8 cache_prev_page = cache_curr_page; 
        cache_curr_page = (cache_curr_page + 1) % 2;

        #pragma unroll(qlen)
        for (u32 qidx = 0; qidx < qlen; ++qidx) {

            // 0: left, 1: up, 2: diag
            u8 parents[3] = { 0 };

            parents[0] = cache[warp_id][cache_prev_page][qidx][lane];
            parents[1] = 0; // No gap penalties at first row
            parents[2] = 0;

            // If we are at the first row
            if (qidx != 0) {
                parents[1] = cache[warp_id][cache_curr_page][qidx - 1][lane];
                parents[2] = cache[warp_id][cache_prev_page][qidx - 1][lane]; 
            }

#if DEBUG
            if (warp_id == 0 && lane == 20) {
                printf("loaded (left: %d, up: %d, diag: %d)\n", 
                    parents[0], parents[1], parents[2]);
            }
#endif

            // Update costs
            
            parents[0] += cost_gaps;
            parents[1] += cost_gaps;
            
            // TODO: change memory layout of strings, have 'lane' last
            parents[2] += (query[qidx] != strings[warp_id][lane][tidx]) ? 1 : 0;

#if DEBUG
            if (warp_id == 0 && lane == 20) {
                printf("update (left: %d, up: %d, diag: %d) for q:%c vs t:%c\n", 
                    parents[0], parents[1], parents[2], query[qidx], strings[warp_id][lane][tidx]);
            }
#endif

            // Find best cell
            u8 best_score = 255;

            #pragma unroll(3)
            for (u32 i = 0; i < 3; ++i)
                if (parents[i] < best_score) {
                    best_score = parents[i];
                }

#if DEBUG
            if (warp_id == 0 && lane == 20) {
                printf("store %d\n", best_score);
            }
#endif

            // Store results
            cache[warp_id][cache_curr_page][qidx][lane] = best_score;
            warp.sync();
        }

        warp.sync();
    }

    // Store the best global score
    result[blockDim.x * blockIdx.x + block.thread_rank()] = cache[warp_id][cache_curr_page][qlen - 1][lane];

#if DEBUG
    printf("score %2d: %d\n", lane, best_global_score);
#endif

}

/*
union reg_mem {
    u32 packed;
    struct {
        u8 a, b, c, d;
    };
};
*/

/// This version uses shared memory only for the query and the strings
/// all tmp values are stored in registers
template<u32 warp_count, u32 qlen, u32 tlen>
__global__ void filter1(u8 *gquery, u8 *gstrings, u8* result, u32 N) {

    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile warp = cg::tiled_partition<warp_size>(block);
    auto warp_id = warp.meta_group_rank();
    auto lane = warp.thread_rank();

    // Cache stored in registers, each thread requires qlen x 2 bytes,
    // that is qlen x 2 / 4 registers of 32 bits each.
    u8 cache[2][qlen];
    u8 cache_curr_page = 0;

    // Keep strings and query in shared memory for fast access
    __shared__ u8 strings[warp_count][warp_size][tlen];
    __shared__ u8 query[qlen];

    // Load strings, each warp load its strings
    #pragma unroll(warp_size)
    for (auto sidx = 0; sidx < warp_size; ++sidx)
        if (lane < tlen) {
            const auto global_idx = (blockDim.x * blockIdx.x * tlen) + (warp_id * warp_size * tlen) + (sidx * tlen); 
            strings[warp_id][sidx][lane] = gstrings[global_idx + lane];
        }

    // Load query for all the warps
    if (warp_id == 0 && lane < qlen)
        query[lane] = gquery[lane];

    // Initialize cache with base values
    #pragma unroll(qlen)
    for (auto qidx = 0; qidx < qlen; ++qidx) {
        cache[0][qidx] = qidx + 1;
    }

#if DEBUG
    block.sync();
    if (warp_id == 0 && lane == 0) {
        
        // Print strings
        printf("strings:\n");
        for (u32 i = 0; i < warp_size; ++i) {
            printf(" %2d ", i);
            for (u32 s = 0; s < tlen; ++s) {
                printf("%c", strings[warp_id][i][s]);
            }
            printf("\n");
        }

        // Print query
        printf("query:\n    ");
        for (u32 s = 0; s < qlen; ++s) {
            printf("%c", query[s]);
        }
        printf("\n");
    }
#endif

    // SAFETY: Guard for strings and query in shared memory
    warp.sync();

    u8 best_global_score = 255;

    #pragma unroll(tlen)
    for (u32 tidx = 0; tidx < tlen; ++tidx) {

        // Advance cache circular buffer
        // NOTE: underflow is OK
        u8 cache_prev_page = cache_curr_page; 
        cache_curr_page = (cache_curr_page + 1) % 2;

        #pragma unroll(qlen)
        for (u32 qidx = 0; qidx < qlen; ++qidx) {

            // 0: left, 1: up, 2: diag
            u8 parents[3] = { 0 };

            // NOTE: Is the index calculation optimized?
            parents[0] = cache[cache_prev_page][qidx];
            parents[1] = 0; // No gap penalties at first row
            parents[2] = 0;

            // If we are at the first row
            if (qidx != 0) {
                parents[1] = cache[cache_curr_page][qidx - 1];
                parents[2] = cache[cache_prev_page][qidx - 1]; 
            }

#if DEBUG
            if (warp_id == 0 && lane == 20) {
                printf("loaded (left: %d, up: %d, diag: %d)\n", 
                    parents[0], parents[1], parents[2]);
            }
#endif

            // Update costs
            
            parents[0] += cost_gaps;
            parents[1] += cost_gaps;
            
            // TODO: change memory layout of strings, have 'lane' last
            parents[2] += (query[qidx] != strings[warp_id][lane][tidx]) ? 1 : 0;

#if DEBUG
            if (warp_id == 0 && lane == 20) {
                printf("update (left: %d, up: %d, diag: %d) for q:%c vs t:%c\n", 
                    parents[0], parents[1], parents[2], query[qidx], strings[warp_id][lane][tidx]);
            }
#endif

            // Find best cell
            u8 best_score = 255;

            #pragma unroll(3)
            for (u32 i = 0; i < 3; ++i)
                if (parents[i] < best_score) {
                    best_score = parents[i];
                }

#if DEBUG
            if (warp_id == 0 && lane == 20) {
                printf("store %d\n", best_score);
            }
#endif

            // Store results
            cache[cache_curr_page][qidx] = best_score;
            //warp.sync();
        }

        //warp.sync();
    }
    
    // Store the best global score
    result[blockDim.x * blockIdx.x + block.thread_rank()] = cache[cache_curr_page][qlen - 1];

#if DEBUG
    printf("score %2d: %d\n", lane, best_global_score);
#endif

}

/// This version uses shared memory only for the query and the strings
/// all tmp values are stored in registers. 
//  It also uses a grid level stride approach for fixed block count.
template<u32 warp_count, u32 qlen, u32 tlen>
__global__ void filter2(u8 *gquery, u8 *gstrings, u8* result, u32 N) { 

    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile warp = cg::tiled_partition<warp_size>(block);
    auto warp_id = warp.meta_group_rank();
    auto lane = warp.thread_rank();

    // Keep strings in shared memory for fast access
    __shared__ u8 strings[warp_count][warp_size][tlen];

    // Cache stored in registers, each thread requires qlen x 2 bytes,
    // that is qlen x 2 / 4 registers of 32 bits each.
    u8 cache[2][qlen];

    // Load query for all the warps in shared memory
    // NOTE: This can go in constant memory
    __shared__ u8 query[qlen];
    if (warp_id == 0 && lane < qlen)
        query[lane] = gquery[lane];

    block.sync();

#if DEBUG_QUERY
        if (warp_id == 0 && lane == 0) {
            // Print query
            printf("query:\n    ");
            for (u32 s = 0; s < qlen; ++s) {
                printf("%c", query[s]);
            }
            printf("\n");
        }
#endif

    auto stride = blockDim.x * gridDim.x;
    for (u32 gidx = blockDim.x * blockIdx.x + threadIdx.x; gidx < N; gidx += stride) {

        // Load strings, each warp load its strings
        #pragma unroll(warp_size)
        for (u8 wsidx = 0; wsidx < warp_size; ++wsidx) {
            const auto read_idx = gidx + (warp_id * warp_size + wsidx) * tlen;
            strings[warp_id][wsidx][lane] = gstrings[read_idx];
        }

        warp.sync();

#if DEBUG_STRINGS
        block.sync();
        if (warp_id == 0 && lane == 0) {
            
            // Print strings
            printf("strings for warp %d:\n", warp_id);
            for (u32 warp_sidx = 0; warp_sidx < warp_size; ++warp_sidx) {
                printf("%2d ", warp_sidx);
                for (u32 s = 0; s < tlen; ++s) {
                    printf("%c", strings[warp_id][warp_sidx][s]);
                }
                printf("\n");
            }
        }
#endif

#if DEBUG == 2
        if (warp_id == 0 && lane == 0) {
            printf("query: ");
            for (u32 qidx = 0; qidx < qlen; ++qidx) {
                printf("%c", query[qidx]);
            }
            printf("\n");
        }
#endif

#if DEBUG == 2
        if (warp_id == 0 && lane == 0) {
            printf("targt: ");
            for (u32 tidx = 0; tidx < tlen; ++tidx) {
                printf("%c", strings[warp_id][lane][tidx]);
            }
            printf("\n");
        }
#endif

        // Initialize cache with base values
        #pragma unroll(qlen)
        for (auto qidx = 0; qidx < qlen; ++qidx) {
            cache[0][qidx] = qidx + 1;
            cache[1][qidx] = 0;
        }

        // SAFETY: Guard for strings and query in shared memory
        warp.sync();

        // Start always on first page of cache
        u8 cache_curr_page = 0;

        #pragma unroll(tlen)
        for (u32 tidx = 0; tidx < tlen; ++tidx) {

            // Advance cache circular buffer
            // NOTE: underflow is OK
            u8 cache_prev_page = cache_curr_page; 
            cache_curr_page = (cache_curr_page + 1) % 2;

            #pragma unroll(qlen)
            for (u32 qidx = 0; qidx < qlen; ++qidx) {

                // 0: left, 1: up, 2: diag
                u8 parents[3] = { 0 };

                // NOTE: Is the index calculation optimized?
                parents[0] = cache[cache_prev_page][qidx];
                parents[1] = 0; // No gap penalties at first row
                parents[2] = 0;

                // If we are at the first row
                if (qidx != 0) {
                    parents[1] = cache[cache_curr_page][qidx - 1];
                    parents[2] = cache[cache_prev_page][qidx - 1]; 
                }

    #if DEBUG == 2
                if (warp_id == 0 && lane == 0) {
                    printf("loaded (left: %d, up: %d, diag: %d)\n", 
                        parents[0], parents[1], parents[2]);
                }
    #endif

                // Update costs
                
                parents[0] += cost_gaps;
                parents[1] += cost_gaps;
                parents[2] += (query[qidx] != strings[warp_id][lane][tidx]) ? 1 : 0;

    #if DEBUG == 2
                if (warp_id == 0 && lane == 0) {
                    printf("update (left: %d, up: %d, diag: %d) for q:%c vs t:%c\n", 
                        parents[0], parents[1], parents[2], query[qidx], strings[warp_id][lane][tidx]);
                }
    #endif

                // Find best cell
                u8 best_score = 255;

                #pragma unroll(3)
                for (u32 i = 0; i < 3; ++i)
                    if (parents[i] < best_score) {
                        best_score = parents[i];
                    }

    #if DEBUG == 2
                if (warp_id == 0 && lane == 0) {
                    printf("store %d\n", best_score);
                }
    #endif

                // Store results
                cache[cache_curr_page][qidx] = best_score;
                //warp.sync();
            }
        }

        // Store the best global score
        result[gidx] = cache[cache_curr_page][qlen - 1];
        //warp.sync();

#if DEBUG
        printf("score %2d: %d\n", lane, result[gidx]);
#endif

    }
}