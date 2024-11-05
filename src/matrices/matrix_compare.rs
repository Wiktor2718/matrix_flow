use super::matrix::{Handle, Matrix};

extern "C" {
    fn matrix_value_type_cmp(a: Handle, b: Handle, len: usize) -> i32;
}

impl PartialEq for Matrix {
    fn eq(&self, other: &Self) -> bool {
        if self.rows != other.rows || self.cols != other.cols {
            return false;
        }
        0 != unsafe {matrix_value_type_cmp(self.handle(), other.handle(), self.len())}
    }
}

#[cfg(test)]
mod tests {
    use crate::prelude::Matrix;

    #[test]
    fn test_eq() {
        let a = Matrix::from([1., 2., 3., 4.], 2, 2);
        let b = Matrix::from([1., 3., 2., 4.], 2, 2);
        assert_ne!(a, b);
        assert_eq!(a, b.t());
    }
}