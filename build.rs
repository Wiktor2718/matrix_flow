use std::process::Command;
use std::env;
use std::fs;

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();

    // Create a directory for the output if it doesn't exist
    fs::create_dir_all(&out_dir).unwrap();

    // Compile the CUDA code into an object file with PIC
    let cuda_arch = "sm_86"; // Adjust the architecture as needed
    let nvcc_status = Command::new("nvcc")
        .args(&["-arch", cuda_arch, "-Xcompiler", "-fPIC", "cuda/main.cu", "-c", "-o"])
        .arg(format!("{}/main.o", out_dir))
        .status()
        .expect("Failed to compile CUDA code");

    if !nvcc_status.success() {
        panic!("nvcc failed to compile the CUDA code");
    }

    // Create a static library from the object file
    let ar_status = Command::new("ar")
        .args(&["crus", "libcuda_kernels.a", "main.o"])
        .current_dir(&out_dir)
        .status()
        .expect("Failed to create static library");

    if !ar_status.success() {
        panic!("ar failed to create the static library");
    }

    // Path to CUDA libraries
    let cuda_lib_path = "/usr/local/cuda/lib64"; // Adjust this path as necessary

    // Link the output directory, CUDA libraries, and relevant CUDA libraries
    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-search=native={}", cuda_lib_path);
    println!("cargo:rustc-link-lib=static=cuda_kernels");
    println!("cargo:rustc-link-lib=dylib=cudart"); // CUDA Runtime API
    println!("cargo:rustc-link-lib=dylib=cublas"); // Link to cuBLAS
}

