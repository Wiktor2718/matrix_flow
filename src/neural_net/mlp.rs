use crate::{matrices::matrix::RM_Handle, prelude::{CudaVec, ValueType}};
use std::sync::Arc;

extern "C" {
    fn normal(res: *mut ValueType, len: usize);
}

pub struct MLP {
    pub data_block: Arc<CudaVec<ValueType>>,
    pub layers: Vec<Layer>,
    pub layers_data: Vec<LayerData>,
    pub learning_rate: ValueType
}

impl MLP {
    pub fn new<T: AsRef<[Layer]>>(batch_size: usize, learning_rate: ValueType, layers: T) -> Self {
        let layers = layers.as_ref();
        let len = Self::calculate_len(layers, batch_size);

        let data_block: Arc<CudaVec<ValueType>> = CudaVec::new(len);
        let base_ptr = data_block.ptr;

        let mut offset = 0;
        let mut layers_data = Vec::new();
        for layer in layers {
            let (d, m, n) = (layer.input_size, layer.output_size, batch_size);
            unsafe {
                normal(base_ptr.add(offset), m*(d + 1)); // init weights and biases
                let weights          = RM_Handle {ptr: base_ptr.add(offset), rows: d, cols: m};
                offset += d * m;
                let biases           = RM_Handle {ptr: base_ptr.add(offset), rows: 1, cols: m};
                offset += 1 * m;
                let weights_gradient = RM_Handle {ptr: base_ptr.add(offset), rows: d, cols: m};
                offset += d * m;
                let biases_gradient  = RM_Handle {ptr: base_ptr.add(offset), rows: 1, cols: m};
                offset += 1 * m;
                let output           = RM_Handle {ptr: base_ptr.add(offset), rows: n, cols: m};
                offset += n * m;
                let output_gradient  = RM_Handle {ptr: base_ptr.add(offset), rows: n, cols: m};
                offset += n * m;
                let input            = RM_Handle {ptr: base_ptr.add(offset), rows: n, cols: d};
                offset += n * d;
                let input_gradient   = RM_Handle {ptr: base_ptr.add(offset), rows: n, cols: d};
                offset += n * d;

                layers_data.push(LayerData {
                    weights,
                    biases,
                    weights_gradient,
                    biases_gradient,
                    output,
                    output_gradient,
                    input,
                    input_gradient,
                });
            }
        }
        Self { data_block, layers: layers.to_vec(), layers_data, learning_rate }
    }

    fn calculate_len(layers: &[Layer], batch_size: usize) -> usize {
        let mut res = 0;
        for layer in layers {
            let (d, m, n) = (layer.input_size, layer.output_size, batch_size);
            res +=  d*m + // W
                    1*m + // B
                    d*m + // dW
                    1*m + // dB
                    n*m + // Y
                    n*m + // dY
                    n*d + // X
                    n*d;  // dX
        }
        res
    }
}

//pub type Layer = (usize, usize, ActivationType);
#[derive(Debug, Clone, Copy)]
pub struct Layer {
    pub input_size: usize,
    pub output_size: usize,
    pub activation: ActivationType,
}

impl Layer {
    pub fn new(input_size: usize, output_size: usize, activation: ActivationType) -> Self {
        Self { input_size, output_size, activation }
    }
}


#[derive(Debug, Clone, Copy)]
pub struct LayerData {
    pub weights: RM_Handle,
    pub biases: RM_Handle,
    pub weights_gradient: RM_Handle,
    pub biases_gradient: RM_Handle,
    pub output: RM_Handle,
    pub output_gradient: RM_Handle,
    pub input: RM_Handle,
    pub input_gradient: RM_Handle,
}

#[derive(Debug, Clone, Copy)]
pub enum ActivationType {
    Linear,
    ReLu,
    Tanh,
    Sigmoid,
    // all possible activations
}

#[cfg(test)]
mod tests {
    use crate::neural_net::mlp::{MLP, ActivationType, Layer};

    #[test]
    fn memory_allocation() {
        let net = MLP::new(2, 0.01, [
            Layer::new(2, 3, ActivationType::ReLu),
            Layer::new(3, 1, ActivationType::Linear)],
        );
        
        println!("{:?}", net.data_block.as_vec());

        println!("{:#?}", net.layers_data);
    }
}