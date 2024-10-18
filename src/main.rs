use matrix_flow::prelude::*;
use std::{iter::zip, time::Instant};

fn mse(y_true: &Matrix, y_pred: &Matrix) -> f32 {
    let t = y_pred - y_true;
    t.sqware().as_vec().iter().sum::<ValueType>() / (t.len() as f32)
}

fn mse_prime(y_true: &Matrix, y_pred: &Matrix) -> Matrix {
    let t = y_pred - y_true;
    2. * t / (y_true.len() as f32)
}

fn print_matrix(matrix: &Matrix) { // It is only for debugging
    for i in 0..matrix.rows {
        for j in 0..matrix.cols {
            print!("{} ", matrix.read_from_index([i, j]));
        }
        println!("");
    }
}

fn main() {
    let start = Instant::now();
    let input_data: Vec<Matrix> = vec![Matrix::from([0., 0.], 2, 1),
                                       Matrix::from([0., 1.], 2, 1),
                                       Matrix::from([1., 0.], 2, 1),
                                       Matrix::from([1., 1.], 2, 1)];

    let output_data: Vec<Matrix> = vec![Matrix::from([0.], 1, 1),
                                        Matrix::from([1.], 1, 1),
                                        Matrix::from([1.], 1, 1),
                                        Matrix::from([0.], 1, 1)];

    let network  = MLP::new(1, 0.001, [
        Layer::new(2,    1000, ActivationType::Tanh),
        Layer::new(1000, 1000, ActivationType::Tanh),
        Layer::new(1000, 1000, ActivationType::Sigmoid),
        Layer::new(1000, 1,    ActivationType::Linear),
    ]);

    const EPOCHS: u32 = 10000;

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

}