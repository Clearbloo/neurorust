use crate::stage::{Predictor, Trainable};

#[derive(Clone, Debug)]
struct DecisionTree {}

impl Trainable for DecisionTree {
    fn train(self) -> Self {
        DecisionTree {}
    }
}

impl Predictor for DecisionTree {
    type Input = i32;
    type Output = i32;
    type Parameters = i32;
    fn parameters(&self) -> &Self::Parameters {
        todo!()
    }
    fn forward(&self, input: &Self::Input) -> Self::Output {
        todo!()
    }
}
