use crate::matrices::{cuda_vec::ValueType, matrix::Matrix};

pub fn mse(y_true: &Matrix, y_pred: &Matrix) -> f32 {
    let t = y_pred - y_true;
    t.sqware().as_vec().iter().sum::<ValueType>() / (t.len() as f32)
}

pub fn mse_prime(y_true: &Matrix, y_pred: &Matrix) -> Matrix {
    let t = y_pred - y_true;
    2. * t / (y_true.len() as f32)
}