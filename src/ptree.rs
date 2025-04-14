use std::{collections::{BTreeSet, HashSet}, fmt::Error, ops::Sub};

use serde::{Deserialize, Serialize};

/// Convert a sequence to an integer preserving lexicographical ordering
fn seq_to_key(seq: &[u8]) -> u64 {
    assert!(2 * seq.len() <= 64, "Cannot reppresent string");
    seq.iter().fold(0u64, |acc, e| {
        let bits = match e {
            b'A' => 0,
            b'C' => 1,
            b'G' => 2,
            b'T' => 3,
            _ => panic!("invalid character!")
        };
        (acc << 2) | bits
    })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PackedTreeInner {
    pub levels: Vec<Vec<u8>>,
    pub offset: Vec<Vec<u32>>
}

/// Generic implementation
impl PackedTreeInner
{
    /// Insert a new sequence, they must be ordered
    pub fn insert_ordered(&mut self, seq: &[u8]) {

        let mut forked = false;
        let mut parent = u32::MAX;
        for (level, e) in seq.iter().enumerate() {
            
            let this_level = &mut self.levels[level];
            let this_ofset = &mut self.offset[level];

            // We forked upstream
            if forked {

                this_level.push(*e);
                this_ofset.push(parent);

                parent = this_level.len() as u32 - 1; 
                continue;
            }

            // We are not the first element in the layer
            if let Some(last) = this_level.last() {
                if e != last { // We are different from the previous sequence
                    
                    this_level.push(*e);
                    this_ofset.push(parent);

                    parent = this_level.len() as u32 - 1;
                    forked = true;
                
                } else { // We are equal to the previous sequence

                    parent = this_level.len() as u32 - 1;
                }
            } else { // Add first element
                this_level.push(*e);
                if level == 0 {
                    this_ofset.push(u32::MAX);
                } else {
                    this_ofset.push(0);
                }
            }
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PackedTree {
    inner: PackedTreeInner,
    depth: usize,
}

impl PackedTree {
    
    /// Create PackedTree from a fasta sequence
    pub fn from_fasta(fasta: &[u8], depth: usize) -> Self {
        let seq_count = fasta.len() - depth + 1;

        // Create a sorting key for all sequences
        let mut sort_keys = vec![u64::MAX; seq_count];
        for (i, seq) in fasta.windows(depth).enumerate() {
            sort_keys[i] = seq_to_key(seq);
        }

        // Sort the indices to obtain the correct sequence order
        let mut indices: Vec<u32> = (0..seq_count as u32).collect();
        indices.sort_unstable_by_key(|&i| sort_keys[i as usize]);

        // Data owner
        let mut inner = PackedTreeInner {
            levels: vec![vec![]; depth],
            offset: vec![vec![]; depth], 
        };

        for beg in indices {
            let beg = beg as usize;
            inner.insert_ordered(&fasta[beg..beg+depth]);
        }

        Self { inner, depth }
    }

    pub fn sequence_at_leaf(&self, index: usize) -> String {
        let mut seq = vec![b'X'; self.depth];

        seq[self.depth - 1] = self.inner.levels[self.depth - 1][index];
        let mut offset = self.inner.offset[self.depth - 1][index] as usize;

        for l in 1..self.depth {
            seq[self.depth - 1 - l] = self.inner.levels[self.depth - 1 - l][offset];
            offset = self.inner.offset[self.depth - 1 - l][offset] as usize;
        }

        return std::str::from_utf8(&seq)
            .unwrap().to_string();
    }

    /// Returns an HashSet with all complete sequences present in the tree
    pub fn sequences(&self) -> HashSet<String> {
        let mut seq_set = HashSet::new();
        for s in 0..self.span() {
            seq_set.insert(self.sequence_at_leaf(s));
        }
        return seq_set;
    }

    /// Returns how many sequences are stored
    pub fn span(&self) -> usize {
        self.inner.levels[self.depth - 1].len()
    }

    /*
    /// Split PackedTree into smaller local PackedTree(s) with byte indices
    pub fn split_at_width(&self, width: usize) -> Vec<PackedTree> {
        let mut results = vec![];
        
        let mut beg = 0;
        let sequence_count = self.span();
        while beg < sequence_count {

            let mut packed_tree = PackedTreeInner {
                levels: vec![vec![]; self.depth],
                offset: vec![vec![]; self.depth],
            };

            // Initial run-lenght and initial minimum offset
            let mut rlen = width;
            let mut rmin = beg;

            // Process each layer in sequence
            for layer in 0..self.depth {

                // Clip the last block if we are out of bounds
                rlen = rlen.min(sequence_count - rmin);

                // Copy layer elements into local tree
                packed_tree.levels[self.depth - 1 - layer].extend_from_slice(
                    &self.inner.levels[self.depth - 1 - layer][rmin..rmin+rlen]);
                
                let global_offsets = &self.inner.offset[self.depth - 1 - layer][rmin..rmin+rlen];
                
                // New minimum index value
                let new_rmin = global_offsets[0];
                let new_rlen = global_offsets[rlen - 1] as usize - rmin + 1;

                // Convert to the specified offset memory type
                // Update local offsets inside the PackedTree structure
                let result = &mut packed_tree.offset[self.depth - 1 - layer];
                for i in rmin..rmin+rlen {
                    
                    // Try to convert from current index to desired index
                    let offset = global_offsets[i];
                    let index: usize = offset as usize - rmin;

                    result[i - rmin] = index.try_into().unwrap_or(0);
                }

                // New layer slice lenght
                rmin = new_rmin as usize;
                rlen = new_rlen;
            }

            results.push(packed_tree);
            beg += width;
        }

        results.into_iter()
            .map(|x| PackedTree { inner: x, depth: self.depth })
            .collect()
    } */

    pub fn alignment_scores(target: &[u8], max_gaps: u8, max_mismatches: u8) -> Vec<(u8, u8)> {
        unimplemented!()
    }

} 

impl std::fmt::Display for PackedTree {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "packed_tree::elements:")?;
        for i in 0..self.depth {
            let letters: Vec<char> = self.inner.levels[i].iter().map(|x| (*x).into()).collect(); 
            writeln!(f, "[{:>4}]: {:?}", i, letters)?;
        }
        writeln!(f, "packed_tree::offsets:")?;
        for i in 0..self.depth {
            writeln!(f, "[{:>4}]: {:?}", i, self.inner.offset[i])?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use crate::utils::{self};
    use super::*;

    const REF_SIZE: usize = 300000;
    const ANCHOR_LEN: usize = 32;
    const RNG_SEED: u64 = 3666;

    /// Tests that the tree contains the correct sequences
    #[test] fn correct_content() {

        // Create random reference sequence
        let reference = utils::generate_test_sequence(REF_SIZE, b"ACTG", RNG_SEED);

        // Insert windows into reference set
        let mut real_seq_set: HashSet<String> = HashSet::new();
        for seq in reference.windows(ANCHOR_LEN) {
            let seq_string = std::str::from_utf8(&seq).unwrap();
            real_seq_set.insert(String::from(seq_string));
        }

        // Check that they contain the same elements
        let packed_tree = PackedTree::from_fasta(&reference, ANCHOR_LEN);
        assert_eq!(packed_tree.sequences(), real_seq_set);
    }

    /// Tests that the tree can be split
    #[test] fn split() {

        // Create random reference sequence
        let reference = utils::generate_test_sequence(REF_SIZE, b"ACTG", RNG_SEED);

        // Insert windows into reference set
        let mut real_seq_set: HashSet<String> = HashSet::new();
        for seq in reference.windows(ANCHOR_LEN) {
            let seq_string = std::str::from_utf8(&seq).unwrap();
            real_seq_set.insert(String::from(seq_string));
        }

        // Split packed tree into multiple local trees and aggregate all sequences
        let packed_tree = PackedTree::from_fasta(&reference, ANCHOR_LEN);
        let split_packed_trees = packed_tree.split_at_width(64);

        let mut split_seq_set: HashSet<String> = HashSet::new();
        for split_tree in &split_packed_trees {
            let split_tree_seqs = split_tree.sequences();
            split_seq_set.extend(split_tree_seqs);
        }

        // Check that they contain the same elements
        assert!(real_seq_set.eq(&split_seq_set));
    }
}