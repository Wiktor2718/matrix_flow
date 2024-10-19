use crate::matrices::{matrix::RM_Handle, cuda_vec::*};
use crate::neural_net::mlp::Layer;
use std::sync::Arc;

pub struct Adam {
    pub data_block: Arc<CudaVec<ValueType>>,
    pub optimizer_data: Vec<AdamData>,
    pub beta1: ValueType,
    pub beta2: ValueType,
    pub epsilon: ValueType,
    pub beta1_t_decayed: ValueType,
    pub beta2_t_decayed: ValueType,
}

impl Adam {
    pub fn new<T: AsRef<[Layer]>>(layers: T, beta1: ValueType, beta2: ValueType, epsilon: ValueType) -> Self {
        let layers = layers.as_ref();
        let len = Self::calculate_len(layers);

        let data_block: Arc<CudaVec<ValueType>> = CudaVec::new(len);
        let base_ptr = data_block.ptr;

        let mut offset = 0;
        let mut optimizer_data = Vec::new();
        for layer in layers {
            let (d, m) = (layer.input_size, layer.output_size);
            unsafe {
                let m_weights = RM_Handle { ptr: base_ptr.add(offset), rows: d, cols: m };
                offset += d * m;
                let v_weights = RM_Handle { ptr: base_ptr.add(offset), rows: d, cols: m };
                offset += d * m;
                let m_biases  = RM_Handle { ptr: base_ptr.add(offset), rows: 1, cols: m };
                offset += 1 * m;
                let v_biases  = RM_Handle { ptr: base_ptr.add(offset), rows: 1, cols: m };
                offset += 1 * m;

                optimizer_data.push(AdamData { m_weights, v_weights, m_biases, v_biases });
            }
        }

        Self {data_block, optimizer_data, beta1, beta2, epsilon, beta1_t_decayed: beta1, beta2_t_decayed: beta2}
    }

    fn calculate_len(layers: &[Layer]) -> usize {
        let mut res = 0;
        for layer in layers {
            let (d, m) = (layer.input_size, layer.output_size);
            res +=  d * m + // mW
                    d * m + // vW
                    1 * m + // mB
                    1 * m;  // vB
        }
        res
    }
}

pub struct AdamData {
    pub m_weights: RM_Handle,
    pub v_weights: RM_Handle,
    pub m_biases: RM_Handle,
    pub v_biases: RM_Handle,
}