use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};


pub fn print_dpt(dpt: &[(u8, u8)], stride: usize) {
    println!();
    let mut i = 0;
    while i < dpt.len() {
        let cell = dpt[i];
        print!("({:2}, {:2}) ", cell.0, cell.1);
        if (i+1) % stride == 0 {
            println!();
        }
        i += 1;
    }
    println!();
}

/// Return the best aligment score possible in term of minimum amount of gaps and mismatches
pub fn mine_score(target: &[u8], sequence: &[u8], max_gaps: u8, max_mismatches: u8) -> bool {
    
    let mut queue: VecDeque<(u8, u8, u8, u8)> = VecDeque::new();
    queue.push_back((0,0,0,0));

    let mut visited: BTreeSet<(u8,u8,u8,u8)> = BTreeSet::new();

    while !queue.is_empty() {

        //println!("queue size: {}", queue.len());
        let (i, j, gaps, mismatches) = queue.pop_back().unwrap();
        
        if visited.contains(&(i,j,gaps,mismatches)) {
            continue;
        } else {
            visited.insert((i,j,gaps,mismatches));
        }

        // Skip invalid
        if gaps > max_gaps || mismatches > max_mismatches {
            continue;
        }

        // End
        if i == target.len() as u8 {
            println!("{:?}, len = {}", visited, visited.len());
            return true;
        }

        // Match or mismatch
        if i < target.len() as u8 && j < sequence.len() as u8 {
            let mismatches = if target[i as usize] != sequence[j as usize] { mismatches + 1 } else { mismatches };
            queue.push_back((i+1, j+1, gaps, mismatches));
        }

        if j < sequence.len() as u8 {
            queue.push_back((i+0, j+1, gaps + 1, mismatches));
        }

        if i < target.len() as u8 {
            queue.push_back((i+1, j+0, gaps + 1, mismatches));
        }
    }

    println!("{:?}, len = {}", visited, visited.len());
    return false;
}