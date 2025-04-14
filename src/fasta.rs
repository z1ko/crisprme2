use std::{io, path::Path};

// A sequence is a vector of bytes
type Seq = Vec<u8>;

pub fn load_from_file<P: AsRef<Path>>(path: P) -> io::Result<Seq> {
    let result: Seq = std::fs::read(path)?;
    Ok(result.into_iter()
        .map(|e| { // Convert to uppercase
            e.to_ascii_uppercase()
        })
        .filter(|byte| // Removes all new-lines and Ns
            *byte != ('\n' as u8) && 
            *byte != ('N'  as u8)
        )
        .collect())
}