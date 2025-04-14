use rand::{Rng, SeedableRng};

/// Create a random reference sequence
pub fn generate_test_sequence(length: usize, nucleotides: &[u8], seed: u64) -> Vec<u8> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let take_one = || nucleotides[rng.random_range(0..nucleotides.len())]; // Uniform random sequences
    std::iter::repeat_with(take_one)
        .take(length).collect()
}