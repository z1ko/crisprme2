use std::collections::HashMap;

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct State {
    i: usize,
    j: usize,
    g: usize,
    m: usize,
}

fn align(
    i: usize, j: usize, g: usize, m: usize,
    s1: &[char], s2: &[char], g_max: usize, m_max: usize,
    memo: &mut HashMap<State, Option<(usize, String, String)>>,
) -> Option<(usize, String, String)> {
    let key = State { i, j, g, m };
    if let Some(val) = memo.get(&key) {
        return val.clone();
    }

    if g > g_max || m > m_max {
        return None;
    }

    if i == s1.len() && j == s2.len() {
        return Some((0, String::new(), String::new()));
    }

    let mut best: Option<(usize, String, String)> = None;

    // Match or mismatch
    if i < s1.len() && j < s2.len() {
        let is_match = s1[i] == s2[j];
        let next = align(
            i + 1,
            j + 1,
            g,
            m + if is_match { 0 } else { 1 },
            s1,
            s2,
            g_max,
            m_max,
            memo,
        );
        if let Some((score, a1, a2)) = next {
            let new_score = score + 1;
            let new_a1 = format!("{}{}", s1[i], a1);
            let new_a2 = format!("{}{}", s2[j], a2);
            best = Some((new_score, new_a1, new_a2));
        }
    }

    // Gap in seq1 (insert in seq2)
    if j < s2.len() {
        let next = align(i, j + 1, g + 1, m, s1, s2, g_max, m_max, memo);
        if let Some((score, a1, a2)) = next {
            let new_score = score + 1;
            let new_a1 = format!("-{}", a1);
            let new_a2 = format!("{}{}", s2[j], a2);
            if best.is_none() || new_score > best.as_ref().unwrap().0 {
                best = Some((new_score, new_a1, new_a2));
            }
        }
    }

    // Gap in seq2 (insert in seq1)
    if i < s1.len() {
        let next = align(i + 1, j, g + 1, m, s1, s2, g_max, m_max, memo);
        if let Some((score, a1, a2)) = next {
            let new_score = score + 1;
            let new_a1 = format!("{}{}", s1[i], a1);
            let new_a2 = format!("-{}", a2);
            if best.is_none() || new_score > best.as_ref().unwrap().0 {
                best = Some((new_score, new_a1, new_a2));
            }
        }
    }

    memo.insert(key, best.clone());
    best
}

pub fn constrained_alignment(query: &str, target: &str, max_gaps: usize, max_miss: usize) -> Option<(usize, String, String)> {
    let mut memo: HashMap<State, Option<(usize, String, String)>> = HashMap::new();
    let s1: Vec<char> = query.chars().collect();
    let s2: Vec<char> = target.chars().collect();
    align(0, 0, 0, 0, &s1, &s2, max_gaps, max_miss, &mut memo)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test] fn alignment() {
        
        let seq1 = "ACGT";
        let seq2 = "ACCT";

        let g_max = 1;
        let m_max = 1;

        match constrained_alignment(seq1, seq2, g_max, m_max) {
            Some((score, a1, a2)) => {
                println!("Alignment score: {}", score);
                println!("Seq1: {}", a1);
                println!("Seq2: {}", a2);
            }
            None => println!("No valid alignment under constraints."),
        }
    }
}