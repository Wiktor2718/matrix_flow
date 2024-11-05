# Matrix Flow
Matrix Flow is a simple machine learning library written in Rust and CUDA.
It was created as a portfolio project to deepen my understanding of machine
learning, GPU programming, and Rust.
It provides an API for matrix manipulation and includes specially optimized neural networks.

## Features
- **GPU-accelerated computation** using CUDA
- **Multi-layer perceptron (MLP)** with customizable layers and activation functions
- **Adam optimizer** for efficient gradient-based learning
- Supports **batch training** for improved efficiency
- **NVTX benchmarking** for performance profiling

## Prerequisites
### Install Rust
- **Rust**: Install Rust from the [official website](https://www.rust-lang.org/).

### Install CUDA Toolkit
- **CUDA Toolkit**: Make sure you have CUDA installed on your system. The default library paths
    are set up for Linux, but you may need to adjust these for your specific environment.

### Install Sample Data Sets
- **CSV Data**: The example provided expects labeled datasets in CSV format for both training and testing.

## Running The Sample MLP
1. Make sure that the GPU architecture in the `build.rs` is correct
    ```rs
    let cuda_arch = "sm_86"; // Adjust the architecture as needed
    ```
1. If you have a non-standard path to CUDA libraries, modify this line  
    ```rs
    let cuda_lib_path = "/usr/local/cuda/lib64"; // Adjust this path as necessary
    ```
1. Run `cargo run`

## Memory Optimizations
This library avoids additional memory allocations by reusing the memory of the
owned operand. For instance:
- `A + &B` reuses the memory of `A`
- `&A + B` reuses the memory of `B`
- `&A + &B` allocates new memory.
- `A + B` reuses the memory of `A` and drops `B`

## Example Usage (Networks)
```rs
use std::{fs::File, iter::zip, path::Path, error::Error};
use matrix_flow::prelude::*;

fn read_labeled_data<P: AsRef<Path>>(path: P, output_size: usize, batch_size: usize, max_value: ValueType) -> Result<(Vec<Matrix>, Vec<Matrix>), Box<dyn Error>> {
    let file = File::open(path)?;
    let mut rdr = csv::Reader::from_reader(file);

    let mut res_items = Vec::new();
    let mut res_labels = Vec::new();

    let mut item_buffer = Vec::new();
    let mut label_buffer = Vec::new();

    for (idx, result) in rdr.deserialize().enumerate() {
        let (label_index, item): (usize, Vec<f32>) = result?;

        // Vectorize label
        let mut label = vec![0.; output_size];
        label[label_index] = 1.;

        // Store in buffers
        item_buffer.extend(&item);
        label_buffer.extend(&label);
        
        // Accumulate until full
        if (idx + 1) % batch_size != 0 {
            continue;
        }

        // Move to GPU
        let mut item_batch = Matrix::from(&item_buffer, batch_size, item.len());
        let label_batch = Matrix::from(&label_buffer, batch_size, output_size);

        // Normalize items on the GPU
        item_batch = item_batch / max_value;

        // Clear buffers
        item_buffer.clear();
        label_buffer.clear();

        // Push batches to results
        res_items.push(item_batch);
        res_labels.push(label_batch);
    }

    Ok((res_labels, res_items))
}

fn main() {
    // Parameters
    const EPOCHS: u32 = 100;
    const BATCH_SIZE: usize = 128;

    let layers = [
        Layer::new(28*28, 100, ActivationType::Tanh),
        Layer::new(100, 100, ActivationType::Tanh),
        Layer::new(100, 10, ActivationType::Linear),
    ];

    range_push("Data Loading");
    let (output_data, input_data) = read_labeled_data(
        "data_sets/mnist_train.csv",
        10,
        BATCH_SIZE,
        255.0
    ).expect("Unable to read data");

    range_pop();

    range_push("Adam Initialization");
    let optim = Optimizer::adam(layers, 0.9, 0.999, 1e-8);
    range_pop();

    range_push("Network Initialization");
    let network = MLP::new(BATCH_SIZE, 0.001, optim, layers);
    range_pop();

    range_push("Training");
    for e in 0..EPOCHS {
        let mut error = 0.;
        for (x, y) in zip(&input_data, &output_data) {
            range_push("Forward Pass");
            let output = network.forward(x);
            range_pop();

            range_push("Error Calculation");
            error += mse(y, &output);
            range_pop();
            
            range_push("Gradient Calculation");
            let gradient = mse_prime(y, &output);
            range_pop();

            range_push("Backward Pass");
            let _ = network.backward(&gradient);
            range_pop();
        }
        println!("{e}: {}", error / input_data.len() as f32);
    }
    range_pop();

    // Free up VRAM space
    drop(output_data);
    drop(input_data);

    let (output_data, input_data) = read_labeled_data(
        "data_sets/mnist_test.csv",
        10,
        128,
        255.
    ).expect("Unable to read data");

    // Test run
    let mut error = 0.0;
    for (x, y) in zip(&input_data, &output_data) {
        let output = network.forward(x);
        error += mse(y, &output);
    }
    println!("Test loss: {}", error / input_data.len() as f32);
}
```

## Important Functions
- `MLP::new(batch_size, learning_rate, optimizer, layers)`
    Creates a MLP struct and allocates its data as a **contiguous block of memory on the GPU (device)**
- `Optimizer::adam(layers, beta1, beta2, epsilon)` creates an `Optimizer` `enum` with the `Adam` variant
    and allocates its data as a **contiguous block of memory on the GPU (device)**
- `forward(batch)` performs a forward pass on a `batch`
- `backward(gradient)` **updates the parameters** with `Optimizer` and **returns its gradient**
- `range_push(label)` and `range_pop()` are used for **NVTX debugging**

## License
This project is licensed under the **MIT License**