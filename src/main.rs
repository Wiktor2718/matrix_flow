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
