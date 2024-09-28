use matrix_flow::prelude::*;

fn main() {
    let a = CudaVec::from(vec![1, 2, 3]);
    println!("{}", a.read_from_index(1));
}
