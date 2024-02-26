use crate::{activation::Activation, layer::DenseLayer, loss::Loss, optimizer::Optimization};
use ndarray::Array2;
use std::sync::Arc;


pub struct Network<L: Loss, O: Optimization> {
    layers: Vec<DenseLayer>,
    loss: L,
    optimizer: O,
    epochs: i32,
}

// impl<A: Activation, L: Loss, O: Optimization> Default for Network<A, L, O> {
//     fn default() -> Self {
//         Self::new(vec![1], MeanAbsoluteError,Adam)
//     }
// }

impl<L: Loss, O: Optimization> Network<L, O> {
    pub fn new(
        architecture: &Vec<usize>,
        activations: &[Arc<dyn Activation>],
        loss: L,
        optimizer: O,
    ) -> Self {
        let mut layers = Vec::new();
        for (idx, &num_neurons) in architecture.iter().enumerate() {
            if idx < architecture.len() - 1 {
                let num_neurons_in_next_layer = architecture[idx + 1];
                // Now we can simply clone the Arc, which is cheap
                let activation = activations[idx].clone();
                layers.push(DenseLayer::new(
                    num_neurons,
                    num_neurons_in_next_layer,
                    activation,
                ));
            }
        }

        Network {
            layers,
            loss,
            optimizer,
            epochs: 100,
        }
    }

    pub fn add_layer(&mut self, layer: DenseLayer) {
        self.layers.push(layer);
    }

    pub fn forward(&mut self, input: &Array2<f64>) -> Array2<f64> {
        self.layers
            .iter_mut()
            .fold(input.clone(), |acc, layer| layer.forward(&acc))
    }

    // This method should implement the logic to perform a backward pass through the network,
    // updating weights and biases based on the gradient of the loss function with respect to the output.
    pub fn backward(&mut self, error: Array2<f64>) {
        println!("{}", error);
        todo!();
    }

    pub fn calculate_loss_gradient(
        &self,
        predictions: &Array2<f64>,
        targets: &Array2<f64>,
    ) -> Array2<f64> {
        self.loss.calculate_gradient(predictions, targets)
    }

    /// Performs backpropagation to update weights and biases of all layers.
    pub fn backwards(&mut self, outputs: &Array2<f64>, targets: &Array2<f64>) {
        let mut inputs_for_layers = Vec::new();

        // Precompute inputs for each layer
        for (i, _) in self.layers.iter().enumerate() {
            inputs_for_layers.push(self.get_input_for_layer(i));
        }

        // Now, iterate over layers with mutable access
        let loss_gradient = self.calculate_loss_gradient(outputs, targets);
        let mut output_gradient: Array2<f64> = loss_gradient; // This should be initialized with the gradient of the loss function w.r.t the output of the last layer.

        // Collect updates for each layer
        let mut weight_updates: Vec<Array2<f64>> = Vec::new();
        let mut bias_updates: Vec<Array2<f64>> = Vec::new();

        for (i, layer) in self.layers.iter_mut().enumerate().rev() {
            let input = &inputs_for_layers[i];
            let (weight_gradient, bias_gradient, input_gradient) = layer.grad_layer(input, &output_gradient);
            weight_updates.push(weight_gradient);
            bias_updates.push(bias_gradient);

            output_gradient = input_gradient; // Prepare for the next iteration
        }

        // Apply collected updates
        self.update_parameters(weight_updates, bias_updates);
            todo!()

    }

    fn update_parameters(&mut self, weight_updates: Vec<Array2<f64>>, bias_updates: Vec<Array2<f64>>) {
        // TODO - Can probably just delete this method
        self.optimizer.apply_updates(weight_updates, bias_updates)
    }

    /// Gets the input for a specific layer.
    /// This is a placeholder for however you decide to implement it.
    fn get_input_for_layer(&self, layer_index: usize) -> Array2<f64> {
        // Return the input Array2<f64> for the specified layer.
        self.layers[layer_index].input.clone()
    }
    

    /// Each loop in epoch, forward pass, calculate loss, backwards pass to calculate gradients
    /// Update parameters (using optimizer), repeat.
    fn train(mut self, input: &Array2<f64>, targets: &Array2<f64>) {
        for _e in 0..self.epochs {
            let outputs = self.forward(input);
            self.backwards(&outputs, targets);
        }
    }
}

#[cfg(test)]
mod test_network {
    use ndarray::arr2;

    use crate::{
        activation::{LeakyReLU, ReLU},
        loss::MeanSquaredError,
        optimizer::Adam,
    };

    use super::*;

    #[test]
    fn test_init() {
        let architecture = vec![1, 2, 1];
        let activations: Vec<Arc<dyn Activation>> = vec![Arc::new(ReLU {}), Arc::new(LeakyReLU {})];
        Network::new(&architecture, &activations, MeanSquaredError, Adam);
    }

    #[test]
    fn test_train() {
        let architecture = vec![1, 2, 1];
        let activations: Vec<Arc<dyn Activation>> = vec![Arc::new(ReLU {}), Arc::new(LeakyReLU {})];
        let net = Network::new(&architecture, &activations, MeanSquaredError, Adam);

        let input = arr2(&[[1.0]]);
        let targets = arr2(&[[2.0]]);

        net.train(&input, &targets)
    }
}
