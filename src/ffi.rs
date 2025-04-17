

/// Connection to CUDA engine
#[cxx::bridge]
mod ffi {
    unsafe extern "C++" {
        include!("crisprme2/include/kernels.hh");

        //type PackedTreeGPU;

        //unsafe fn device_allocate(data: *mut u8, levels: *mut u32, depth: u32) -> UniquePtr<PackedTreeGPU>;

        // Test print value
        unsafe fn mine(data: *mut u8, n: i32);
    }
}