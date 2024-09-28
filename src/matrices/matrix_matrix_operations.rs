use std::ops::{Add, Sub, Mul, Div};
use super::matrix::{Matrix, Handle};

extern "C" {
    fn cudaDeviceSynchronize();

    fn dot(res: Handle, a: Handle, b: Handle);

    fn matrix_add_t_right(res: Handle, a: Handle, b: Handle, len: usize);
    fn matrix_sub_t_right(res: Handle, a: Handle, b: Handle, len: usize);
    fn matrix_mul_t_right(res: Handle, a: Handle, b: Handle, len: usize);
    fn matrix_div_t_right(res: Handle, a: Handle, b: Handle, len: usize);

    fn matrix_sub_t_left(res: Handle, a: Handle, b: Handle, len: usize);
    fn matrix_div_t_left(res: Handle, a: Handle, b: Handle, len: usize);
}

// result uses the same memoy layout as left matrix
macro_rules! impl_operation_t_right {
    ($trait:ident, $function:ident, $call:ident) => {
        impl $trait<Matrix> for Matrix {
            type Output = Matrix;
            fn $function(self, rhs: Matrix) -> Self::Output {
                debug_assert!(self.rows == rhs.rows);
                debug_assert!(self.cols == rhs.cols);
                unsafe {
                    $call(self.handle(), self.handle(), rhs.handle(), self.len());
                    cudaDeviceSynchronize();
                }
                self
            }
        }

        impl $trait<&Matrix> for Matrix {
            type Output = Matrix;
            fn $function(self, rhs: &Matrix) -> Self::Output {
                debug_assert!(self.rows == rhs.rows);
                debug_assert!(self.cols == rhs.cols);
                unsafe {
                    $call(self.handle(), self.handle(), rhs.handle(), self.len());
                    cudaDeviceSynchronize();
                }
                self
            }
        }

        impl $trait<&Matrix> for &Matrix {
            type Output = Matrix;
            fn $function(self, rhs: &Matrix) -> Self::Output {
                debug_assert!(self.rows == rhs.rows);
                debug_assert!(self.cols == rhs.cols);
                let res = self.empty_clone();
                unsafe {
                    $call(res.handle(), self.handle(), rhs.handle(), self.len());
                    cudaDeviceSynchronize();
                }
                res
            }
        }
    };
}

// result uses the same memory layout as right matrix (reusing commutative functions)
macro_rules! impl_operation_commutatve {
    ($trait:ident, $function:ident, $call:ident) => {
        impl $trait<Matrix> for &Matrix {
            type Output = Matrix;
            fn $function(self, rhs: Matrix) -> Self::Output {
                debug_assert!(self.rows == rhs.rows);
                debug_assert!(self.cols == rhs.cols);
                unsafe {
                    $call(rhs.handle(), rhs.handle(), self.handle(), self.len());
                    cudaDeviceSynchronize();
                }
                rhs
            }
        }
    };
}

// result uses the same memory layout as right matrix (dedicated functions)
macro_rules! impl_operation_t_left {
    ($trait:ident, $function:ident, $call:ident) => {
        impl $trait<Matrix> for &Matrix {
            type Output = Matrix;
            fn $function(self, rhs: Matrix) -> Self::Output {
                debug_assert!(self.rows == rhs.rows);
                debug_assert!(self.cols == rhs.cols);
                unsafe {
                    $call(rhs.handle(), self.handle(), rhs.handle(), self.len());
                    cudaDeviceSynchronize();
                }
                rhs
            }
        }
    };
}

impl_operation_t_right!(Add, add, matrix_add_t_right);
impl_operation_t_right!(Sub, sub, matrix_sub_t_right);
impl_operation_t_right!(Mul, mul, matrix_mul_t_right);
impl_operation_t_right!(Div, div, matrix_div_t_right);

impl_operation_commutatve!(Add, add, matrix_add_t_right);
impl_operation_commutatve!(Mul, mul, matrix_mul_t_right);

impl_operation_t_left!(Sub, sub, matrix_sub_t_left);
impl_operation_t_left!(Div, div, matrix_div_t_left);

impl Matrix {
    pub fn dot(&self, other: &Self) -> Self {
        let res = Matrix::new(self.rows, other.cols);
        unsafe {
            dot(res.handle(), self.handle(), other.handle());
            cudaDeviceSynchronize();
        }
        res
    }
}

#[cfg(test)]
mod tests {
    use crate::matrices::matrix::Matrix;

    #[test]
    fn test_dot() {
        let a = Matrix::from(vec![1., 2., 3., 4.], 2, 2);
        let b = Matrix::from(vec![1., 2., 3., 4.], 2, 2);
        assert!(a.dot(&b.t()).as_vec() == vec![5., 11., 11., 25.]);
    }

    #[test]
    fn test_add_flat() {
        let a = Matrix::from(vec![1., 2., 3., 4.], 2, 2);
        let b = Matrix::from(vec![4., 3., 2., 1.], 2, 2);
        assert!((&a + &b).as_vec() == vec![5.; 4]);
        assert!((&a.t() + &b.t()).as_vec() == vec![5.; 4]);
    }

    #[test]
    fn test_sub_flat() {
        let a = Matrix::from(vec![2., 3., 4., 5.], 2, 2);
        let b = Matrix::from(vec![1., 2., 3., 4.], 2, 2);
        assert!((&a - &b).as_vec() == vec![1.; 4]);
        assert!((&a.t() - &b.t()).as_vec() == vec![1.; 4]);
    }

    #[test]
    fn test_mul_flat() {
        let a = Matrix::from(vec![2., 3., 2., 3.], 2, 2);
        let b = Matrix::from(vec![3., 2., 3., 2.], 2, 2);
        assert!((&a * &b).as_vec() == vec![6.; 4]);
        assert!((&a.t() * &b.t()).as_vec() == vec![6.; 4]);
    }

    #[test]
    fn test_div_flat() {
        let a = Matrix::from(vec![12., 6., 6., 12.], 2, 2);
        let b = Matrix::from(vec![6., 3., 3., 6.], 2, 2);
        assert!((&a / &b).as_vec() == vec![2.; 4]);
        assert!((&a.t() / &b.t()).as_vec() == vec![2.; 4]);
    }

    #[test]
    fn test_add_t() {
        let a = Matrix::from(vec![1., 2., 3., 4.], 2, 2);
        let b = Matrix::from(vec![2., 0., 1., -1.], 2, 2);
        println!("{:?}", (&a + &b.t()).as_vec());
        assert!((&a + &b.t()).as_vec() == vec![3.; 4]);
        assert!((&a.t() + &b).as_vec() == vec![3.; 4]);
    }
    #[test]
    fn test_add_t_left() {
        let a = Matrix::from(vec![1., 2., 3., 4.], 2, 2);
        let b = Matrix::from(vec![2., 0., 1., -1.], 2, 2);
        assert!((&a.t() + b).as_vec() == vec![3.; 4]);
    }
}