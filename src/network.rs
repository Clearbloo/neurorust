use crate::layer::DenseLayer; // Assuming the layer file is named layer.rs and contains DenseLayer
use ndarray::ArrayD;

pub struct Network {
    layers: Vec<DenseLayer>,
    // Optionally, include properties for loss functions and optimizers
}

impl Network {
    pub fn new() -> Self {
        Network {
            layers: Vec::new(),
            // Initialize other properties as needed
        }
    }

    pub fn add_layer(&mut self, layer: DenseLayer) {
        self.layers.push(layer);
    }

    pub fn forward(&self, input: ArrayD<f64>) -> ArrayD<f64> {
        self.layers.iter().fold(input, |acc, layer| layer.forward(acc))
    }

    // This method should implement the logic to perform a backward pass through the network,
    // updating weights and biases based on the gradient of the loss function with respect to the output.
    pub fn backward(&mut self, error: ArrayD<f64>) {
        println!("{}", error);
        todo!();
    }

    // Implement other methods as needed, such as for training, evaluating the model, etc.
}
