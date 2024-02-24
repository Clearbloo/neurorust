use crate::{activation::Activation, layer::DenseLayer};
use ndarray::Array2;

pub struct Network<A: Activation> {
    layers: Vec<DenseLayer<A>>,
    // Properties for loss functions and optimizers
}

impl<A: Activation> Default for Network<A> {
    fn default() -> Self {
        Self::new()
    }
}

impl<A: Activation> Network<A> {
    pub fn new() -> Self {
        Network {
            layers: Vec::new(),
            // Initialize other properties as needed
        }
    }

    pub fn add_layer(&mut self, layer: DenseLayer<A>) {
        self.layers.push(layer);
    }

    pub fn forward(&self, input: Array2<f64>) -> Array2<f64> {
        self.layers
            .iter()
            .fold(input, |acc, layer| layer.forward(acc))
    }

    // This method should implement the logic to perform a backward pass through the network,
    // updating weights and biases based on the gradient of the loss function with respect to the output.
    pub fn backward(&mut self, error: Array2<f64>) {
        println!("{}", error);
        todo!();
    }

    // Implement other methods as needed, such as for training, evaluating the model, etc.
}
