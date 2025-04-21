use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read, Write};
use std::{collections::HashSet, io};

use ptree::PackedTree;

mod fasta;
mod ptree;
mod utils;
mod align;

//use ptree::Tree;

const SEQ_LEN: usize = 20;

/// Connection to CUDA engine
#[cxx::bridge]
mod ffi {
    unsafe extern "C++" {
        include!("crisprme2/include/kernels.hh");

        // Test print value
        unsafe fn mine(data: *mut u8, n: i32);

        unsafe fn filter(query: *const u8, trgts: *const u8, result: *mut u8, qlen: i32, tlen: i32, n: i32);
    }
}

fn main() -> io::Result<()> {

    tests::launch_kernel_fasta();
    return Ok(());

    let fasta = &fasta::load_from_file("fasta/chr22.fa")?;
    let seq_count = fasta.len() - SEQ_LEN + 1;
    //println!("fasta: {:?}", std::str::from_utf8(fasta).unwrap());
    println!("fasta seq. count: {}", seq_count);
    
    let tree = PackedTree::from_fasta(&fasta, SEQ_LEN);
    println!("Unique sequences: {}", tree.span());

    // Store tree on the disk
    let bytes = bincode::serialize(&tree).unwrap();
    let output_path = std::path::Path::new("data").join("chr22.ptree");
    let mut file = std::fs::File::create(output_path).unwrap();
    file.write_all(&bytes).unwrap();

    //println!("How many unique sequences? {}", sequences.len());
    //println!("How much memory to store them in this form? {:.1} MB", sequences.len() as f32 * 16.0 / 1e6);
    //println!("How much memory to store them in the optimal prefix-tree case? {:.1} MB", tree_cnt(sequences.len()) as f32 / 1e6 * 2.0);

    //let mut data: Vec<u8> = vec![0];
    //unsafe {
    //    ffi::mine(data.as_mut_ptr(), data.len() as i32);
    //}

    Ok(())
}

//#[cfg(test)]
mod tests {
    use crate::utils::{generate_test_sequence};
    use super::*;

    pub(crate) fn launch_kernel_fasta() {

        // Size of the target sequences
        const TLEN: usize = 24;
        // Size of the query sequence
        const QLEN: usize = 24;
        // Shift between sequences
        const DELTA: usize = 1;
        // Edit distance threshold
        const THRESHOLD: u8 = 0;
        // Number of sequences
        const N: usize = 40000000;

        // Create random reference sequence
        let fasta = fasta::load_from_file("/home/z1ko/univr/parco/crisprme2/fasta/chr22.fa").unwrap();
        println!("fasta size: {} bases", fasta.len());
        
        // Add all strings into linear memory
        let mut targets: Vec<u8> = Vec::new();

        let mut beg  = 0;
        let mut n = 0;
        while beg < fasta.len() - TLEN - 1 && n < N {
            targets.extend(&fasta[beg..beg+TLEN]);

            beg += DELTA;
            n += 1;
        }

        assert_eq!(targets.len(), n * TLEN);
        println!("sequence count: {} ({} MB)", n, (n * TLEN) as f32 / 1e6);
        let mut results = vec![255; n];
        
        /*
        println!("data: ");
        for i in 0..n {
            let beg = i * TLEN;
            let seq = std::str::from_utf8(&targets[beg..beg+TLEN]).unwrap();
            println!("{:>2} {}", i, seq);
        }
         */

        // Random query string
        let query = utils::generate_test_sequence(QLEN, b"ACTG", 2025);
        let query = targets[0..QLEN].to_owned();
        println!("query: {}", std::str::from_utf8(&query).unwrap());

        unsafe {
            ffi::filter(
                query.as_ptr(), 
                targets.as_ptr(),
                results.as_mut_ptr(), 
                QLEN as i32, 
                TLEN as i32, 
                n    as i32
            );
        }

        // Results
        //println!("results: {:?}", results);

        // How many elements below threshold?
        let valid = results.iter().map(|e| if *e <= THRESHOLD { 1 } else { 0 }).sum::<u32>();
        println!("valid results: {:?}", valid);
    }
}