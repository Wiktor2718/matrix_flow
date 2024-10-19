use matrix_flow::{neural_net::{adam::Adam, mlp::Optimizer}, prelude::*};
use std::{iter::zip, time::Instant};

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
    let input_data: Vec<Matrix> = vec![Matrix::from([0., 0.], 2, 1),
                                       Matrix::from([0., 1.], 2, 1),
                                       Matrix::from([1., 0.], 2, 1),
                                       Matrix::from([1., 1.], 2, 1)];

    let output_data: Vec<Matrix> = vec![Matrix::from([0.], 1, 1),
                                        Matrix::from([1.], 1, 1),
                                        Matrix::from([1.], 1, 1),
                                        Matrix::from([0.], 1, 1)];
    let layers = [
        Layer::new(2,    100, ActivationType::Tanh),
        Layer::new(100, 100, ActivationType::Tanh),
        Layer::new(100, 100, ActivationType::Sigmoid),
        Layer::new(100, 1,    ActivationType::Linear),
    ];
    let optim = Optimizer::Adam(Adam::new(layers, 0.9, 0.999, 1e-7));
    let network  = MLP::new(1, 0.001, optim, layers);

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