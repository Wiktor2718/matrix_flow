use crate::{matrices::matrix::{Matrix, RM_Handle}, neural_net::mlp::MLP, prelude::memcpy};
use crate::matrices::cuda_vec::*;
use super::mlp::ActivationType;

extern "C" {
    fn forward_mlp(x: RM_Handle, w: RM_Handle, b: RM_Handle, y: RM_Handle);
    fn forward_leaky_relu(y: RM_Handle, x: RM_Handle);
    fn forward_sigmoid(y: RM_Handle, x: RM_Handle);
    fn forward_tanh(y: RM_Handle, x: RM_Handle);

    fn cudaDeviceSynchronize();
}

impl MLP {
    pub fn forward(&self, batch: &Matrix) -> Matrix {
        let last_layer = self.layers_data.last().unwrap();
        let res = Matrix::new(last_layer.output.rows, last_layer.output.cols);

        self.batch_to_mlp(batch);

        for (idx, layer) in self.layers.iter().enumerate() {
            let data = self.layers_data[idx];
            let next_data_handle = self.layers_data
                .get(idx + 1)
                .map_or(res.row_major_handle(), |layer| layer.input);

            unsafe {
                forward_mlp(data.input, data.weights, data.biases, data.output);
                cudaDeviceSynchronize();
            }

            match layer.activation {
                ActivationType::Linear => {
                    if idx == self.layers.len() - 1 {
                        self.output_from_mlp(&res);
                    }
                }
                ActivationType::ReLu => unsafe {
                    forward_leaky_relu(next_data_handle, data.output);
                    cudaDeviceSynchronize();
                }
                ActivationType::Sigmoid => unsafe {
                    forward_sigmoid(next_data_handle, data.output);
                    cudaDeviceSynchronize();
                }
                ActivationType::Tanh => unsafe {
                    forward_tanh(next_data_handle, data.output);
                    cudaDeviceSynchronize();
                }
            }
        }

        res
    }

    fn batch_to_mlp(&self, batch: &Matrix) {
        let err = memcpy(
            batch.data.ptr,
            self.layers_data[0].input.ptr,
            batch.len(),
            cudaMemcpyKind::cudaMemcpyDeviceToDevice,
        );
        if let Err(e) = err {
            panic!("matrix_to_mlp: data copy error: {}", error_string(e));
        }
    }

    fn output_from_mlp(&self, res: &Matrix) {
        let last_layer = self.layers_data.last().unwrap();

        let err = memcpy(
            last_layer.output.ptr,
            res.data.ptr,
            res.len(),
            cudaMemcpyKind::cudaMemcpyDeviceToDevice,
        );
        if let Err(e) = err {
            panic!("output_from_mlp: data copy error: {}", error_string(e));
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{neural_net::mlp::{ActivationType, MLP, Layer, Optimizer}, prelude::Matrix};

    #[test]
    fn forward() {
        let net = MLP::new(2, 0.01, Optimizer::SGD, [
            Layer::new(2, 3, ActivationType::ReLu),
            Layer::new(3, 1, ActivationType::Linear)],
        );

        let batch = Matrix::from([1., 2., 3., 4.], 2, 2);
        println!("{:?}", net.forward(&batch).as_vec());
        println!("{:?}", net.data_block.as_vec());
    }

}