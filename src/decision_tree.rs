use ndarray::Array2;

use crate::model::{Predictor, Trainable};

#[derive(Clone, Debug)]
struct DecisionTree {}

impl Predictor for DecisionTree {
    type Input = Array2<f64>;
    type Output = Array2<f64>;
    fn predict(&self, _inputs: &Self::Input) -> Self::Output {
        todo!()
    }
}
impl Trainable for DecisionTree {
    type Data = Array2<f64>;
    type Target = Array2<f64>;

    fn train(&mut self, _inputs: &Self::Data, _targets: &Self::Target) -> impl Predictor {
        self.clone()
    }
}
