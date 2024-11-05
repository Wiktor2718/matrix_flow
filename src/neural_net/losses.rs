use crate::{matrices::{cuda_vec::ValueType, matrix::Matrix}, prelude::RM_Handle};

extern "C" {
    fn mse_prime_launcher(res: RM_Handle, y_true: RM_Handle, y_pred: RM_Handle);
    fn compute_mse(y_true: RM_Handle, y_pred: RM_Handle) -> ValueType;
}

pub fn mse(y_true: &Matrix, y_pred: &Matrix) -> f32 {
    unsafe {
        compute_mse(
            y_true.row_major_handle(),
            y_pred.row_major_handle(),
        )
    }
}

pub fn mse_prime(y_true: &Matrix, y_pred: &Matrix) -> Matrix {
    let res = y_pred.empty_clone();
    unsafe {
        mse_prime_launcher(
            res.row_major_handle(),
            y_true.row_major_handle(),
            y_pred.row_major_handle(),
        );
    }
    res
}