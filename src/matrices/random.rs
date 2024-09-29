use super::{cuda_vec::ValueType, matrix::Matrix};

extern "C" {
    fn normal(res: *mut ValueType, len: usize);
    fn uniform(res: *mut ValueType, len: usize);
}

impl Matrix {
    pub fn normal(rows: usize, cols: usize) -> Self {
        let res = Matrix::new(rows, cols);
        unsafe { normal(res.data.ptr, res.len()); }
        res
    }

    pub fn uniform(rows: usize, cols: usize) -> Self {
        let res = Matrix::new(rows, cols);
        unsafe { uniform(res.data.ptr, res.len()); }
        res
    }
}