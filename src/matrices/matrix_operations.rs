use std::ops::Neg;
use super::matrix::Matrix;
use super::cuda_vec::*;

extern "C" {
    fn cudaDeviceSynchronize();

    fn matrix_neg(res: *mut ValueType, matrix: *const ValueType, len: usize);

    fn relu(res: *mut ValueType, matrix: *const ValueType, len: usize);
    fn relu_prime(res: *mut ValueType, matrix: *const ValueType, len: usize);
    fn sqware(res: *mut ValueType, matrix: *const ValueType, len: usize);
}

impl Neg for Matrix {
    type Output = Matrix;
    fn neg(self) -> Self::Output {
        unsafe {
            matrix_neg(self.data.ptr, self.data.ptr, self.len());
            cudaDeviceSynchronize();
        }
        self
    } 
}

impl Neg for &Matrix {
    type Output = Matrix;
    fn neg(self) -> Self::Output {
        let res = self.empty_clone();
        unsafe {
            matrix_neg(res.data.ptr, self.data.ptr, self.len());
            cudaDeviceSynchronize();
        }
        res
    } 
}

impl Matrix {
    pub fn relu(&self) -> Self {
        let res = self.empty_clone();
        unsafe {
            relu(res.data.ptr, self.data.ptr, self.len());
            cudaDeviceSynchronize();
        }
        res
    }

    pub fn relu_prime(&self) -> Self {
        let res = self.empty_clone();
        unsafe {
            relu_prime(res.data.ptr, self.data.ptr, self.len());
            cudaDeviceSynchronize();
        }
        res
    }

    pub fn sqware(&self) -> Self {
        let res = self.empty_clone();
        unsafe {
            sqware(res.data.ptr, self.data.ptr, self.len());
            cudaDeviceSynchronize();
        }
        res
    }
}