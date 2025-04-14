// Specify the desired architecture version
const ARCH: &str = "compute_86";
const CODE: &str = "sm_86";

fn main() {

    // Set bridge file for Rust <-> C++ comunication
    let mut cc = cxx_build::bridge("src/main.rs");

    // Setup CUDA
    let cc = cc.cuda(true).std("c++17")
        .flag("-gencode")
        .flag(format!("arch={},code={}", ARCH, CODE))
        .flag("-cudart=shared");
        
    // Link CCCL and compile kernels
    cc.file("src/kernels.cu")
        .includes(&[ // NVIDIA CCCL
            "extern/cccl/libcudacxx/include",
            "extern/cccl/thrust",
            "extern/cccl/cub"
        ]) 
        .compile("crisprme2cuda.a");

    // Link CUDA runtime (libcudart.so)
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=cudart");

    //println!("cargo:rerun-if-changed=src/main.rs");
    println!("cargo:rerun-if-changed=src/kernels.cc");
    println!("cargo:rerun-if-changed=include/kernels.hh");
}