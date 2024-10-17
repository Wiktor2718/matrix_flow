use crate::{matrices::matrix::{Matrix, RM_Handle}, neural_net::mlp::MLP, prelude::memcpy};
use crate::matrices::cuda_vec::*;
use super::mlp::ActivationType;

extern "C" {
    fn backward_mlp(x: RM_Handle, w: RM_Handle, d_y: RM_Handle, d_w: RM_Handle, d_b: RM_Handle, d_x: RM_Handle);
    fn backward_leaky_relu(d_y: RM_Handle, x: RM_Handle, d_x: RM_Handle);

    fn update_params(w: RM_Handle, d_w: RM_Handle, b: RM_Handle, d_b: RM_Handle, learning_rate: ValueType);

    fn cudaDeviceSynchronize();
}

impl MLP {
    pub fn backward(&self, grad: &Matrix) -> Matrix {
        for (idx, layer) in self.layers.iter().enumerate().rev() {
            let data = self.layers_data[idx];
            let next_data_handle = self.layers_data
                .get(idx + 1)
                .map_or(grad.row_major_handle(), |layer| layer.input_gradient);

            match layer.2 {
                ActivationType::Linear => unsafe {
                    if idx == self.layers.len() - 1 {
                        self.grad_to_mlp(grad);
                    }
                    backward_mlp(
                        data.input,
                        data.weights,
                        data.output_gradient,
                        data.weights_gradient,
                        data.biases_gradient,
                        data.input_gradient,
                    );
                    cudaDeviceSynchronize();
                    update_params(
                        data.weights,
                        data.weights_gradient,
                        data.biases,
                        data.biases_gradient,
                        0.001,
                    );
                    cudaDeviceSynchronize();
                }
                ActivationType::ReLu => unsafe {
                    backward_leaky_relu(next_data_handle, data.output, data.output_gradient);
                    cudaDeviceSynchronize();
                    backward_mlp(
                        data.input,
                        data.weights,
                        data.output_gradient,
                        data.weights_gradient,
                        data.biases_gradient,
                        data.input_gradient,
                    );
                    cudaDeviceSynchronize();
                    update_params(
                        data.weights,
                        data.weights_gradient,
                        data.biases,
                        data.biases_gradient,
                        0.001,
                    );
                    cudaDeviceSynchronize();
                }
                ActivationType::Sigmoid => todo!("implement sigmoid"),
                ActivationType::Tanh => todo!("implement tanh"), 
            }
        }

        self.grad_from_mlp()
    }

    fn grad_to_mlp(&self, grad: &Matrix) {
        let last_layer = self.layers_data.last().unwrap();
        let err = memcpy(
            grad.data.ptr,
            last_layer.output_gradient.ptr,
            grad.len(),
            cudaMemcpyKind::cudaMemcpyDeviceToDevice,
        );
        if let Err(e) = err {
            panic!("matrix_to_mlp: data copy error: {}", error_string(e));
        }
    }

    fn grad_from_mlp(&self) -> Matrix {
        let mlp_grad = self.layers_data[0].input_gradient;
        let res = Matrix::new(mlp_grad.rows, mlp_grad.cols);

        let err = memcpy(
            mlp_grad.ptr,
            res.data.ptr,
            res.len(),
            cudaMemcpyKind::cudaMemcpyDeviceToDevice,
        );
        if let Err(e) = err {
            panic!("grad_from_mlp: data copy error: {}", error_string(e));
        }

        res
    }
}

#[cfg(test)]
mod tests {
    use crate::{neural_net::mlp::{ActivationType, MLP}, prelude::Matrix};

    #[test]
    fn forward() {
        let net = MLP::new([
            (2, 3, ActivationType::ReLu),
            (3, 1, ActivationType::ReLu)],
            2);

        let loss_prime = Matrix::from([3., 4.], 1, 2);
        println!("{:?}", net.backward(&loss_prime).as_vec());
        println!("{:?}", net.data_block.as_vec());
    }

}