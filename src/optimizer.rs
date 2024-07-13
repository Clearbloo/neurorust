use ndarray::Array2;

use crate::layer::DenseLayer;
// TODO - Also should make into Enums
pub trait Optimization {
    fn apply_updates(
        &self,
        layers: &mut Vec<DenseLayer>,
        weight_updates: &[Array2<f64>],
        bias_updates: &[Array2<f64>],
    );
}

pub struct Adam {
    pub lr: f64,
}
pub struct SGD {
    pub lr: f64,
}

impl Optimization for Adam {
    fn apply_updates(
        &self,
        _layers: &mut Vec<DenseLayer>,
        _weight_updates: &[Array2<f64>],
        _bias_updates: &[Array2<f64>],
    ) {
        // todo!()
    }
}

impl Optimization for SGD {
    fn apply_updates(
        &self,
        layers: &mut Vec<DenseLayer>,
        weight_updates: &[Array2<f64>],
        bias_updates: &[Array2<f64>],
    ) {
        // TODO - Consider normalizing the weight update
        for (i, layer) in layers.iter_mut().rev().enumerate() {
            let layer_weight_update = self.lr * &weight_updates[i];

            layer.weights.data += &layer_weight_update;

            println!("All bias updates {bias_updates:?}");

            let layer_bias_update = self.lr * &bias_updates[i];

            layer.biases.data += &layer_bias_update;
        }
    }
}
