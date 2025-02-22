use crate::model::{Predictor, Trainable};
use crate::utils::Matrix;

#[derive(Clone, Debug)]
struct DecisionTree {}

impl Predictor for DecisionTree {
    type Input = Matrix;
    type Output = Matrix;
    fn predict(&self, _inputs: &Self::Input) -> Self::Output {
        todo!()
    }
}
impl Trainable for DecisionTree {
    type Data = Matrix;
    type Target = Matrix;

    fn train(&mut self, _inputs: &Self::Data, _targets: &Self::Target) -> impl Predictor {
        self.clone()
    }
}
