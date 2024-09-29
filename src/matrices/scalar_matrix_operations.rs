use std::ops::{Add, Sub, Mul, Div};
use super::matrix::Matrix;
use super::cuda_vec::*;

extern "C" {
    fn cudaDeviceSynchronize();

    fn scalar_matrix_sub(res: *mut ValueType, a: ValueType, b: *const ValueType, len: usize);
    fn scalar_matrix_div(res: *mut ValueType, b: ValueType, b: *const ValueType, len: usize);

    // these kernels are reused (commutative)
    fn matrix_scalar_add(res: *mut ValueType, a: *const ValueType, b: ValueType, len: usize);
    fn matrix_scalar_mul(res: *mut ValueType, a: *const ValueType, b: ValueType, len: usize);
}

macro_rules! commutative_scalar_matrix {
    ($trait:ident, $function:ident, $call:ident) => {
        impl $trait<Matrix> for ValueType {
            type Output = Matrix;
            fn $function(self, rhs: Matrix) -> Self::Output {
                unsafe {
                    $call(rhs.data.ptr, rhs.data.ptr, self, rhs.len());
                    cudaDeviceSynchronize();
                }
                rhs
            }
        }

        impl $trait<&Matrix> for ValueType {
            type Output = Matrix;
            fn $function(self, rhs: &Matrix) -> Self::Output {
                let res = rhs.empty_clone();
                unsafe {
                    $call(res.data.ptr, rhs.data.ptr, self, rhs.len());
                    cudaDeviceSynchronize();
                }
                res
            }
        }
    };
}

commutative_scalar_matrix!(Add, add, matrix_scalar_add);
commutative_scalar_matrix!(Mul, mul, matrix_scalar_mul);

macro_rules! impl_scalar_matrix {
    ($trait:ident, $function:ident, $call:ident) => {
        impl $trait<Matrix> for ValueType {
            type Output = Matrix;
            fn $function(self, rhs: Matrix) -> Self::Output {
                unsafe {
                    $call(rhs.data.ptr, self, rhs.data.ptr, rhs.len());
                    cudaDeviceSynchronize();
                }
                rhs
            }
        }

        impl $trait<&Matrix> for ValueType {
            type Output = Matrix;
            fn $function(self, rhs: &Matrix) -> Self::Output {
                let res = rhs.empty_clone();
                unsafe {
                    $call(res.data.ptr, self, rhs.data.ptr, res.len());
                    cudaDeviceSynchronize();
                }
                res
            }
        }
    };
}

impl_scalar_matrix!(Sub, sub, scalar_matrix_sub);
impl_scalar_matrix!(Div, div, scalar_matrix_div);

#[cfg(test)]
mod tests {
    use crate::matrices::matrix::Matrix;

    #[test]
    fn test_add() {
        let mut a = Matrix::from(vec![0.; 100], 10, 10);
        for _ in 0..10 {
            a = 1. + a;
        }
        assert_eq!(a.data.as_vec(), vec![10.; 100]);
    }

    #[test]
    fn test_sub() {
        let mut a = Matrix::from(vec![1.; 100], 10, 10);
        a = 0. - a;
        assert_eq!(a.data.as_vec(), vec![-1.; 100]);
        a = 0. - a;
        assert_eq!(a.data.as_vec(), vec![1.; 100]);
    }

    #[test]
    fn test_mul() {
        let mut a = Matrix::from(vec![1.; 100], 10, 10);
        for _ in 0..10 {
            a = 2. * a;
        }
        assert_eq!(a.data.as_vec(), vec![(2 as f32).powi(10); 100]);
    }

    #[test]
    fn test_div() {
        let mut a = Matrix::from(vec![10.; 100], 10, 10);
        a = 10. / a; 
        assert_eq!(a.data.as_vec(), vec![1.; 100]);
        a = 10. / a; 
        assert_eq!(a.data.as_vec(), vec![10.; 100]);
    }
}