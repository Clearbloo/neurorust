use ndarray::Array2;

use crate::layer::DenseLayer;

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
        weight_updates: &[Array2<f64>],
        bias_updates: &[Array2<f64>],
    ) {
        println!("Weight updates: {:?}", weight_updates);
        println!("Bias updates: {:?}", bias_updates);
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
        println!("Optimizing");
        // TODO - Consider normalizing the weight update
        for (i, layer) in layers.iter_mut().rev().enumerate() {
            let wupdate = self.lr * &weight_updates[i];
            println!("Weight upd: {:?}", wupdate);
            println!("Original layer weights {:?}", layer.weights.data);
            layer.weights.data += &wupdate;
            println!("New layer weights {:?}", layer.weights.data);

            let bupdate = self.lr * &bias_updates[i];
            println!("Bias upd: {:?}", bupdate);
            println!("Original layer Bias {:?}", layer.biases.data);
            layer.biases.data += &bupdate;
            println!("New layer Bias {:?}", layer.biases.data);

        }
    }
}
