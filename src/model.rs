pub trait Trainable {
    type Data;
    type Target;
    fn train(&mut self, inputs: &Self::Data, targets: &Self::Target) -> impl Predictor;
}

pub trait Predictor {
    type Input;
    type Output;
    fn predict(&self, inputs: &Self::Input) -> Self::Output;
}
