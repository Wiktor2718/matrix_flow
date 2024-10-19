use super::{matrix::Matrix, cuda_vec::ValueType};
use std::fmt;

impl fmt::Debug for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let host_data = self.as_vec();

        if self.rows == 0 || self.cols == 0 {
            return write!(f, "Matrix ({}x{}): []", self.rows, self.cols);
        }

        if f.alternate() {
            writeln!(f, "Matrix ({}x{}):", self.rows, self.cols)?;
            for i in 0..self.rows {
                write!(f, "    [")?;
                for j in 0..self.cols {
                    write!(f, "{:.2}", read(&host_data, (i, j), self))?;
                    if j < self.cols - 1 {
                        write!(f, ", ")?;
                    }
                }
                writeln!(f, "]")?;
            }
        } else {
            write!(f, "Matrix ({}x{}): [", self.rows, self.cols)?;
            for i in 0..self.rows {
                write!(f, "[")?;
                for j in 0..self.cols {
                    write!(f, "{:.2}", read(&host_data, (i, j), self))?;
                    if j < self.cols - 1 {
                        write!(f, ", ")?;
                    }
                }

                write!(f, "]")?;
                if i < self.rows - 1 {
                    write!(f, ", ")?;
                }
            }
            write!(f, "]")?;
        }
        Ok(())
    }
}

fn read(host_data: &[ValueType], idx: (usize, usize), device_matrix: &Matrix) -> ValueType {
    let flat_index = if device_matrix.row_major {
        idx.0 * device_matrix.cols + idx.1
    } else {
        idx.0 + device_matrix.rows * idx.1
    };
    host_data[flat_index]
}

#[cfg(test)]
mod tests {
    use crate::prelude::Matrix;

    #[test]
    fn debug_test() {
        let a = Matrix::from([1., 2., 3., 4., 5., 6.], 2, 3);
        let b = Matrix::new(0, 0);
        println!("{:?}", a);
        println!("{:#?}", a);

        println!("{:?}", b);
        println!("{:#?}", b);
    }
}