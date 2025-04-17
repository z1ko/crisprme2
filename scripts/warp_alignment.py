from dataclasses import dataclass
from collections import deque
import random

def main():
    print("Hello from scripts!")

@dataclass
class Node:
    q: int # query index
    t: int # target index
    g: int # gaps
    m: int # mismatches


#         warp_thread: 1 2 3 4 5 6 7 8 9 ...       
# warp_frontier(warp): N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N 
# 
# The stack has a warp_stack_top pointer, implemented as a thread register, all the threads read going back from the pointer,
# and write going forward
#
# A warp read up to (warp_size * K) elements from the top of the stack, each thread can generate up to 3 new stack elements
# The new stack elements are inserted in order of 'closeness' to a solution, first all 'match-mismatch' nodes, that are processed last,
# and then all 'gap' nodes, that are processed immediately next frontier. 
#


def alignment_pop_stack(query: str, target: str, max_gaps: int, max_mismatches: int, warp_size: int) -> tuple[bool, Node, int, int]:

    queue = []
    queue.append(Node(0, 0, 0, 0))

    frontier_sizes = []

    expansions = 0
    while len(queue) != 0:
        #print("frontier before threads size: ", len(queue))
        
        frontier_size = len(queue)
        frontier_sizes.append(frontier_size)
        expansions += 1

        # =======================================================
        # Simulate up to <warp_size> threads

        # Read value
        warp_curr = [None] * warp_size
        for lane_id in range(warp_size):
            if frontier_size - lane_id > 0:
                warp_curr[lane_id] = queue.pop()

        # Check valid
        for lane_id in range(warp_size):
            if frontier_size - lane_id > 0:
                curr = warp_curr[lane_id]

                # Skip invalid nodes
                if (curr.q >= len(query) or curr.t >= len(target) or curr.g > max_gaps or curr.m > max_mismatches):
                    warp_curr[lane_id] = None

                # Found an acceptable solution
                if (curr.q == len(query) - 1):
                    return True, curr, max(frontier_sizes), expansions

        # Add all mismatches
        for lane_id in range(warp_size):
            if frontier_size - lane_id > 0 and warp_curr[lane_id] is not None:
                curr = warp_curr[lane_id]

                mismatches = curr.m + 1 if query[curr.q] != target[curr.t] else curr.m
                queue.append(Node(curr.q + 1, curr.t + 1, curr.g, mismatches))

        # Add gap to target, semi-global
        for lane_id in range(warp_size):
            if frontier_size - lane_id > 0 and warp_curr[lane_id] is not None:
                curr = warp_curr[lane_id]

                t_gaps = curr.g + 1 if curr.q != 0 else curr.g
                queue.append(Node(curr.q, curr.t + 1, t_gaps, curr.m))

        # Add gap to query
        for lane_id in range(warp_size):
            if frontier_size - lane_id > 0 and warp_curr[lane_id] is not None:
                curr = warp_curr[lane_id]
                
                queue.append(Node(curr.q + 1, curr.t, curr.g + 1, curr.m))

    return False, None, max(frontier_sizes), expansions

def alignment_pop_stack_immediate(query: str, target: str, max_gaps: int, max_mismatches: int, warp_size: int) -> tuple[bool, Node, int, int]:

    queue = []
    queue.append(Node(0, 0, 0, 0))

    frontier_sizes = []

    expansions = 0
    while len(queue) != 0:
        #print("frontier before threads size: ", len(queue))
        
        frontier_size = len(queue)
        frontier_sizes.append(frontier_size)
        expansions += 1

        # =======================================================
        # Simulate up to <warp_size> threads

        # Read value
        warp_curr = [None] * warp_size
        for lane_id in range(warp_size):
            if frontier_size - lane_id > 0:
                warp_curr[lane_id] = queue.pop()

        # Check valid
        # Load from shared memory
        #for lane_id in range(warp_size):
        #    if frontier_size - lane_id > 0:
        #        curr = warp_curr[lane_id]
        #
        #        # Skip invalid nodes
        #        if (curr.q >= len(query) or curr.t >= len(target) or curr.g > max_gaps or curr.m > max_mismatches):
        #            warp_curr[lane_id] = None
        #
        #        # Found an acceptable solution
        #        if (curr.q == len(query) - 1):
        #            return True, curr, max(frontier_sizes), expansions

        # Add all mismatches
        # Parallel prefix-scan using warp intrinsics
        # Set warp_write_ptr to sum
        for lane_id in range(warp_size):
            if frontier_size - lane_id > 0 and warp_curr[lane_id] is not None:
                curr = warp_curr[lane_id]

                mismatches = curr.m + 1 if query[curr.q] != target[curr.t] else curr.m
                new_node = Node(curr.q + 1, curr.t + 1, curr.g, mismatches)

                # Check if new node is invalid, dont add to queue
                if (new_node.q >= len(query) or new_node.t >= len(target) or new_node.g > max_gaps or new_node.m > max_mismatches):
                    continue

                # Check if new node is a solution, if so return
                if (new_node.q == len(query) - 1):
                    return True, curr, max(frontier_sizes), expansions

                # Otherwise add to queue
                queue.append(new_node)

        # Add gap to target, semi-global
        for lane_id in range(warp_size):
            if frontier_size - lane_id > 0 and warp_curr[lane_id] is not None:
                curr = warp_curr[lane_id]

                t_gaps = curr.g + 1 if curr.q != 0 else curr.g
                new_node = Node(curr.q, curr.t + 1, t_gaps, curr.m)

                # Check if new node is invalid, dont add to queue
                if (new_node.q >= len(query) or new_node.t >= len(target) or new_node.g > max_gaps or new_node.m > max_mismatches):
                    continue

                # Check if new node is a solution, if so return
                if (new_node.q == len(query) - 1):
                    return True, curr, max(frontier_sizes), expansions

                # Otherwise add to queue
                queue.append(new_node)

        # Add gap to query
        for lane_id in range(warp_size):
            if frontier_size - lane_id > 0 and warp_curr[lane_id] is not None:
                curr = warp_curr[lane_id]
                
                new_node = Node(curr.q + 1, curr.t, curr.g + 1, curr.m)

                 # Check if new node is invalid, dont add to queue
                if (new_node.q >= len(query) or new_node.t >= len(target) or new_node.g > max_gaps or new_node.m > max_mismatches):
                    continue

                # Check if new node is a solution, if so return
                if (new_node.q == len(query) - 1):
                    return True, curr, max(frontier_sizes), expansions

                # Otherwise add to queue
                queue.append(new_node)

    return False, None, max(frontier_sizes), expansions

def random_string(alphabet: str, n: int) -> str:
    return ''.join(random.choice(alphabet) for _ in range(n))

if __name__ == "__main__":

    frontiers = []
    for i in range(1000):

        query  = random_string("ACTG", 24)
        target = random_string("ACTG", 32)

        print(f"alignment of {query} (len={len(query)}) against {target} (len = {len(target)})")
        result, node, max_frontier_size, expansions = alignment_pop_stack_immediate(query, target, 3, 3, 32)
        print(f"result = {result}, max_frontier_size = {max_frontier_size}, expansions = {expansions}")

        frontiers.append(max_frontier_size)
    
    print(f"global max_frontier_size = {max(frontiers)}, expansions = {expansions}")

    result, node, max_frontier_size, expansions = alignment_pop_stack("GATTACAGATTACA", "GATTACAGATTACAGATTACAGATTACA", 1, 0, 32)
    print(f"result = {result}")
