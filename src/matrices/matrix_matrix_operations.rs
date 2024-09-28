use super::matrix::{Matrix, Handle};

extern "C" {
    fn dot(res: Handle, a: Handle, b: Handle);
    fn cudaDeviceSynchronize();
}

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
}