extern crate libc;

pub use crate::matrices::cuda_vec::*;
pub use crate::matrices::nvtx::*;
pub use crate::matrices::matrix::{Matrix, Handle, RM_Handle};

pub use crate::neural_net::mlp::{ActivationType, MLP, LayerData, Layer};