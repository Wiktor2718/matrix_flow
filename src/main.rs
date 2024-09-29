use matrix_flow::prelude::*;
use std::iter::zip;

trait Propagation {
    fn forward(&mut self, input: Matrix) -> Matrix;
    fn backward(&mut self, output_gradient: Matrix, learning_rate: ValueType) -> Matrix;
}

struct Dense {
    weights: Matrix,
    biases: Matrix, input: Matrix,
}

impl Dense {
    fn new(input_size: usize, output_size: usize) -> Self {
        Self {
            weights: Matrix::normal(output_size, input_size),
            //weights: Matrix::from(vec![0.1; output_size * input_size], [output_size, input_size]),
            biases: Matrix::normal(output_size, 1),
            //biases: Matrix::from(vec![0.1; output_size], [output_size, 1]),
            input: Matrix::new(input_size, 1),
        }
    }
}

impl Propagation for Dense {
    fn forward(&mut self, input: Matrix) -> Matrix {
        self.input = input;
        self.weights.dot(&self.input) + &self.biases
    }

    fn backward(&mut self, output_gradient: Matrix, learning_rate: ValueType) -> Matrix {
        let weights_gradient = output_gradient.dot(&self.input.t());

        self.weights = &self.weights - weights_gradient * learning_rate;
        self.biases = &self.biases - &output_gradient * learning_rate;

        self.weights.t().dot(&output_gradient)
    }
}

struct Activation {
    input: Matrix,
}

impl Activation {
    fn new(input_size: usize) -> Self {
        Self { input: Matrix::new(input_size, 1) }
    }
}

impl Propagation for Activation {
    fn forward(&mut self, input: Matrix) -> Matrix {
        self.input = input;
        self.input.relu()
    }

    fn backward(&mut self, output_gradient: Matrix, _learning_rate: ValueType) -> Matrix {
        output_gradient * self.input.relu_prime()
    }
}

fn mse(y_true: &Matrix, y_pred: &Matrix) -> f32 {
    let t = y_pred - y_true;
    t.sqware().as_vec().iter().sum::<ValueType>() / (t.len() as f32)
}

fn mse_prime(y_true: &Matrix, y_pred: &Matrix) -> Matrix {
    let t = y_pred - y_true;
    2. * t / (y_true.len() as f32)
}

fn print_2d_matrix(matrix: &Matrix) { // It is only for debugging
    for i in 0..matrix.rows {
        for j in 0..matrix.cols {
            print!("{} ", matrix.read_from_index([i, j]));
        }
        println!("");
    }
}

fn main() {
    let input_data: Vec<Matrix> = vec![Matrix::from(vec![0., 0.], 2, 1),
                                       Matrix::from(vec![0., 1.], 2, 1),
                                       Matrix::from(vec![1., 0.], 2, 1),
                                       Matrix::from(vec![1., 1.], 2, 1)];

    let output_data: Vec<Matrix> = vec![Matrix::from(vec![0.], 1, 1),
                                        Matrix::from(vec![1.], 1, 1),
                                        Matrix::from(vec![1.], 1, 1),
                                        Matrix::from(vec![0.], 1, 1)];

    let mut network: Vec<Box<dyn Propagation>> = vec![Box::new(Dense::new(2, 1000)),
                                                      Box::new(Activation::new(1000)),
                                                      Box::new(Dense::new(1000, 1))];

    const EPOCHS: u32 = 10000;
    const LEARNING_RATE: f32 = 0.001;

    for e in 0..EPOCHS {
        let mut error: f32 = 0.;
        for (x, y) in zip(&input_data, &output_data) {
            let mut output = x.clone();
            for layer in network.iter_mut() {
                output = layer.forward(output);
            }

            error += mse(y, &output);

            let mut gradient = mse_prime(y, &output);
            for layer in network.iter_mut().rev() {
                gradient = layer.backward(gradient, LEARNING_RATE);
            }
        }
        println!("{e}: {}", error / input_data.len() as f32);
    }

}