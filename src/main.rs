use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read, Write};
use std::{collections::HashSet, io};

use ptree::PackedTree;

mod fasta;
mod ptree;
mod utils;

//use ptree::Tree;

const SEQ_LEN: usize = 24;

/// Connection to CUDA engine
#[cxx::bridge]
mod ffi {
    unsafe extern "C++" {
        include!("crisprme2/include/kernels.hh");

        // Test print value
        unsafe fn mine(data: *mut u8, n: i32);
    }
}

fn main() -> io::Result<()> {

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
