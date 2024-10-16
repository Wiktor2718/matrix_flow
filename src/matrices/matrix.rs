use super::cuda_vec::*;
use std::sync::Arc;

pub struct Matrix {
    pub data: Arc<CudaVec<ValueType>>,
    pub rows: usize,
    pub cols: usize,
    pub row_major: bool,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        let data = CudaVec::new(rows * cols);
        Self { data, rows, cols, row_major: true }
    }

    pub fn from<T: AsRef<[ValueType]>>(data: T, rows: usize, cols: usize) -> Self {
        let data = CudaVec::from(data.as_ref());
        Self { data, rows, cols, row_major: true }
    }

    pub fn empty_clone(&self) -> Self {
        Self {
            data: CudaVec::new(self.data.len),
            rows: self.rows,
            cols: self.cols,
            row_major: self.row_major,
        }
    }

    pub fn clone(&self) -> Self {
        Self {
            data: self.data.copy(),
            rows: self.rows,
            cols: self.cols,
            row_major: self.row_major,
        }
    }

    pub fn as_vec(&self) -> Vec<ValueType> {
        self.data.as_vec()
    }

    pub fn t(&self) -> Self {
        Self {
            data: Arc::clone(&self.data),
            rows: self.cols,
            cols: self.rows,
            row_major: !self.row_major,
        }
    }

    pub fn len(&self) -> usize {
        self.data.len
    }

    pub fn read_from_index<I: AsRef<[usize]>>(&self, index: I) -> ValueType {
        let index = index.as_ref();
        debug_assert_eq!(index.len(), 2, "Index length must be equal to 2");
        debug_assert!(index[0] < self.rows, "rows out of bounds");
        debug_assert!(index[1] < self.cols, "cols out of bounds");

        let flat_index = if self.row_major {
            index[0] * self.cols + index[1]
        } else {
            index[0] + self.rows * index[1]
        };
        
        self.data.read_from_index(flat_index)
    }

    pub fn handle(&self) -> Handle {
        Handle {
            ptr: self.data.ptr,
            rows: self.rows,
            cols: self.cols,
            row_major: if self.row_major { 1 } else { 0 },
        }
    }

    pub fn row_major_handle(&self) -> RM_Handle {
        assert!(self.row_major, "RM_Handle: Matrix is not row-major");
        RM_Handle {
            ptr: self.data.ptr,
            rows: self.rows,
            cols: self.cols,
        }
    }
}

#[repr(C)] #[derive(Clone, Copy, Debug)]
pub struct Handle {
    pub ptr: *mut ValueType,
    pub rows: usize,
    pub cols: usize,
    pub row_major: u8,
}

#[repr(C)] #[derive(Clone, Copy, Debug)]
pub struct RM_Handle {
    pub ptr: *mut ValueType,
    pub rows: usize,
    pub cols: usize,
}