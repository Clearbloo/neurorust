// Rename this module to train or training

// Indicates that a struct can be trained.
// Maybe redundant?
pub trait Trainable: Predictor {
    fn train(self) -> Self;
}

// Indicates that this model can be used to make predictions
// May or may not also be Trainable
pub trait Predictor {
    type Input;
    type Output;
    type Parameters;
    fn parameters(&self) -> &Self::Parameters;
    fn forward(&self, input: &Self::Input) -> Self::Output;
}
