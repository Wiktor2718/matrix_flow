use std::ops::{Add, Sub, Mul, Div};
use super::matrix::Matrix;
use super::cuda_vec::*;

extern "C" {
    fn cudaDeviceSynchronize();

    fn matrix_scalar_add(res: *mut ValueType, a: *const ValueType, b: ValueType, len: usize);
    fn matrix_scalar_sub(res: *mut ValueType, a: *const ValueType, b: ValueType, len: usize);
    fn matrix_scalar_mul(res: *mut ValueType, a: *const ValueType, b: ValueType, len: usize);
    fn matrix_scalar_div(res: *mut ValueType, a: *const ValueType, b: ValueType, len: usize);
}

macro_rules! impl_matrix_scalar {
    ($trait:ident, $function:ident, $call:ident) => {
        impl $trait<ValueType> for Matrix {
            type Output = Matrix;
            fn $function(self, rhs: ValueType) -> Self::Output {
                unsafe {
                    $call(self.data.ptr, self.data.ptr, rhs, self.len());
                    cudaDeviceSynchronize();
                }
                self
            }
        }

        impl $trait<ValueType> for &Matrix {
            type Output = Matrix;
            fn $function(self, rhs: ValueType) -> Self::Output {
                let res = self.empty_clone();
                unsafe {
                    $call(res.data.ptr, self.data.ptr, rhs, self.len());
                    cudaDeviceSynchronize();
                }
                res
            }
        }
    };
}

impl_matrix_scalar!(Add, add, matrix_scalar_add);
impl_matrix_scalar!(Sub, sub, matrix_scalar_sub);
impl_matrix_scalar!(Mul, mul, matrix_scalar_mul);
impl_matrix_scalar!(Div, div, matrix_scalar_div);

#[cfg(test)]
mod tests {
    use crate::matrices::matrix::Matrix;

    #[test]
    fn test_add() {
        let mut a = Matrix::from(vec![0.; 100], 10, 10);
        for _ in 0..10 {
            a = a.t() + 1.;
        }
        assert_eq!(a.data.as_vec(), vec![10.; 100]);
    }

    #[test]
    fn test_sub() {
        let mut a = Matrix::from(vec![10.; 100], 10, 10);
        for _ in 0..10 {
            a = a - 1.;
        }
        assert_eq!(a.data.as_vec(), vec![0.; 100]);
    }

    #[test]
    fn test_mul() {
        let mut a = Matrix::from(vec![1.; 100], 10, 10);
        for _ in 0..10 {
            a = a * 2.;
        }
        assert_eq!(a.data.as_vec(), vec![(2 as f32).powi(10); 100]);
    }

    #[test]
    fn test_div() {
        let mut a = Matrix::from(vec![1024.; 100], 10, 10);
        for _ in 0..10 {
            a = a / 2.;
        }
        assert_eq!(a.data.as_vec(), vec![1.; 100]);
    }
}