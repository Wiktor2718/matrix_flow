use matrix_flow::{neural_net::{adam::Adam, mlp::Optimizer}, prelude::*};
use std::{fs::File, iter::zip, path::Path, time::Instant};
use std::{io, error::Error};

fn read_labeled_data<P: AsRef<Path>>(path: P, output_size: usize, batch_size: usize) -> Result<(Vec<Matrix>, Vec<Matrix>), Box<dyn Error>> {
    let file = File::open(path)?;
    let mut rdr = csv::Reader::from_reader(file);

    let mut res_items: Vec<Matrix> = Vec::new();
    let mut res_labels: Vec<Matrix> = Vec::new();

    let mut item_buffer: Vec<ValueType> = Vec::new();
    let mut label_buffer: Vec<ValueType> = Vec::new();

    for (idx, result) in rdr.deserialize().enumerate() {
        let (label_index, item): (usize, Vec<f32>) = result?;

        // vectorize label
        let mut label = vec![0.; output_size];
        label[label_index] = 1.;

        // store in buffers
        item_buffer.extend(&item);
        label_buffer.extend(&label);
        
        // acumulates until full
        if (idx + 1) % batch_size != 0 {
            continue; 
        }

        // move to gpu
        let item_batch = Matrix::from(&item_buffer, batch_size, item.len());
        let label_batch = Matrix::from(&label_buffer, batch_size, output_size);

        // clearning the bufers
        item_buffer.clear();
        label_buffer.clear();

        // pushing to return 
        res_items.push(item_batch);
        res_labels.push(label_batch);
    }

    Ok((res_labels, res_items))
}

fn mse(y_true: &Matrix, y_pred: &Matrix) -> f32 {
    let t = y_pred - y_true;
    t.sqware().as_vec().iter().sum::<ValueType>() / (t.len() as f32)
}

fn mse_prime(y_true: &Matrix, y_pred: &Matrix) -> Matrix {
    let t = y_pred - y_true;
    2. * t / (y_true.len() as f32)
}

fn main() {
    let start = Instant::now();
    let (output_data, input_data) = read_labeled_data("data_sets/mnist_train.csv", 10, 128).expect("Can't read");

    let layers = [
        Layer::new(28*28,  100, ActivationType::Tanh),
        Layer::new(100, 100, ActivationType::Tanh),
        Layer::new(100, 10,    ActivationType::Linear),
    ];
    let optim = Optimizer::SGD;
    let network  = MLP::new(128, 0.001, optim, layers);

    // train pass
    const EPOCHS: u32 = 100;

    for e in 0..EPOCHS {
        let mut error: f32 = 0.;
        for (x, y) in zip(&input_data, &output_data) {
            let output = network.forward(x);

            error += mse(y, &output);

            let gradient = mse_prime(y, &output);
            let _ = network.backward(&gradient);
        }
        println!("{e}: {}", error / input_data.len() as f32);
    }
    println!("it took {:?}", start.elapsed());

    // save space in vram
    drop(output_data);
    drop(input_data);

    let (output_data, input_data) = read_labeled_data("data_sets/mnist_test.csv", 10, 128).expect("Can't read");

    // test run
    let mut error: f32 = 0.;
    for (x, y) in zip(&input_data, &output_data) {
        let output = network.forward(x);
        error += mse(y, &output);
    }
    println!("test loss: {}", error / input_data.len() as f32);
}