use ndarray::{arr2, Array2};

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
            let wupdate = self.lr * &weight_updates[i];
            println!(
                "Layer number {i},\n wupdate:\n {},\n weights:\n {}",
                wupdate, layer.weights.data
            );
            layer.weights.data += &wupdate;

            println!("All bias updates {:?}", bias_updates);

            let bupdate = self.lr * &bias_updates[i];

            println!(
                "Layer number {i},\n bupdates:\n {:?},\n bias:\n {}",
                bupdate, layer.biases.data
            );
            layer.biases.data += &bupdate;
        }
    }
}
