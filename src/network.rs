use crate::{activation::Activate, layer::Dense, loss::Grad, optimizer::Optimization};
use ndarray::Array2;
use std::sync::Arc;

#[derive(Clone)]
pub struct Network<L: Grad, O: Optimization> {
    layers: Vec<Dense>,
    loss: L,
    optimizer: O,
    epochs: i32,
}

// impl<A: Activate, L: Loss, O: Optimization> Default for Network<A, L, O> {
//     fn default() -> Self {
//         Self::new(vec![1], MeanAbsoluteError,Adam)
//     }
// }

impl<L: Grad, O: Optimization> Network<L, O> {
    pub fn new(
        architecture: &Vec<usize>,
        activations: &[Arc<dyn Activate>],
        loss: L,
        optimizer: O,
    ) -> Self {
        let mut layers = Vec::new();
        for (idx, &num_neurons) in architecture.iter().enumerate() {
            if idx < architecture.len() - 1 {
                let num_neurons_in_next_layer = architecture[idx + 1];
                // Now we can simply clone the Arc, which is cheap
                let activation = activations[idx].clone();
                layers.push(Dense::new(
                    num_neurons,
                    num_neurons_in_next_layer,
                    activation,
                ));
            }
        }

        Self {
            layers,
            loss,
            optimizer,
            epochs: 50,
        }
    }

    pub fn add_layer(&mut self, layer: Dense) {
        self.layers.push(layer);
    }

    pub fn forward(&mut self, input: &Array2<f64>) -> Array2<f64> {
        self.layers
            .iter_mut()
            .fold(input.clone(), |acc, layer| layer.forward(&acc))
    }

    pub fn calculate_loss(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> f64 {
        self.loss.calculate_loss(predictions, targets)
    }

    // This method should implement the logic to perform a backward pass through the network,
    // updating weights and biases based on the gradient of the loss function with respect to the output.
    pub fn calculate_loss_gradient(
        &self,
        predictions: &Array2<f64>,
        targets: &Array2<f64>,
    ) -> Array2<f64> {
        self.loss.calculate_gradient(predictions, targets)
    }

    /// Performs backpropagation to update weights and biases of all layers.
    pub fn backwards(
        &mut self,
        outputs: &Array2<f64>,
        targets: &Array2<f64>,
    ) -> (Vec<Array2<f64>>, Vec<Array2<f64>>) {
        // Now, iterate over layers with mutable access
        let loss_gradient = self.calculate_loss_gradient(outputs, targets);

        // This should be initialized with the gradient of the loss function w.r.t the output of the last layer.
        let mut output_gradient: Array2<f64> = loss_gradient;

        // Collect updates for each layer
        let mut weight_updates: Vec<Array2<f64>> = Vec::new();
        let mut bias_updates: Vec<Array2<f64>> = Vec::new();

        for layer in self.layers.iter_mut().rev() {
            let (weight_gradient, bias_gradient, input_gradient) =
                layer.grad_layer(&output_gradient);
            weight_updates.push(weight_gradient);
            bias_updates.push(bias_gradient);

            output_gradient = input_gradient; // Prepare for the next iteration
        }

        (weight_updates, bias_updates)
    }

    fn update_parameters(
        &mut self,
        weight_updates: &[Array2<f64>],
        bias_updates: &[Array2<f64>],
    ) -> &mut Self {
        // TODO - Can probably just delete this method and use the optimizer directly
        self.optimizer
            .apply_updates(&mut self.layers, weight_updates, bias_updates);
        self
    }

    /// Gets the input for a specific layer.
    /// This is a placeholder for however you decide to implement it.
    pub fn get_input_for_layer(&self, layer_index: usize) -> Array2<f64> {
        // Return the input Array2<f64> for the specified layer.
        self.layers[layer_index].input.clone()
    }

    /// Each loop in epoch, forward pass, calculate loss, backwards pass to calculate gradients
    /// Update parameters (using optimizer), repeat.
    pub fn train(&mut self, input: &Array2<f64>, targets: &Array2<f64>) {
        for _e in 0..self.epochs {
            let outputs = self.forward(input);
            let (weight_updates, bias_updates) = self.backwards(&outputs, targets);
            self.update_parameters(&weight_updates, &bias_updates);
        }
    }

    pub fn get_params(self) {
        let mut weights: Vec<Array2<f64>> = vec![];
        for layer in self.layers {
            let mut w = vec![layer.weights.data];
            weights.append(&mut w);
        }
    }

    pub fn load_params(self, params: Vec<Array2<f64>>) {
        for (i, mut layer) in self.layers.into_iter().enumerate() {
            layer.weights.data = params[i].clone()
        }
    }
}

#[cfg(test)]
mod test_network {
    use ndarray::arr2;

    use crate::{
        activation::Activation,
        loss::Metric,
        optimizer::{Adam, SGD},
    };

    use super::*;

    #[test]
    fn test_init() {
        let architecture_1 = vec![1, 2, 1];
        let activations: Vec<Arc<dyn Activate>> = vec![
            Arc::new(Activation::ReLU),
            Arc::new(Activation::LeakyReLU(0.1)),
        ];
        let net1 = Network::new(
            &architecture_1,
            &activations,
            Metric::MSE,
            Adam { lr: 0.001 },
        );

        assert_eq!(net1.layers.len(), 2);

        let architecture_2 = vec![1, 1];
        let activations_2: Vec<Arc<dyn Activate>> = vec![Arc::new(Activation::ReLU)];
        let net2 = Network::new(
            &architecture_2,
            &activations_2,
            Metric::MSE,
            Adam { lr: 0.001 },
        );

        assert_eq!(net2.layers.len(), 1);
    }

    #[test]
    fn test_train() {
        let architecture = vec![1, 2, 1];
        let activations: Vec<Arc<dyn Activate>> = vec![
            Arc::new(Activation::ReLU),
            Arc::new(Activation::LeakyReLU(0.1)),
        ];
        let mut net = Network::new(&architecture, &activations, Metric::MSE, Adam { lr: 0.001 });

        let input = arr2(&[[1.0]]);
        let targets = arr2(&[[2.0]]);

        net.train(&input, &targets);
    }

    #[test]
    fn test_train_updates_network_outputs() {
        use std::env;
        env::set_var("RUST_BACKTRACE", "1");
        let architecture = vec![2, 2, 1];
        let activations: Vec<Arc<dyn Activate>> = vec![
            Arc::new(Activation::ReLU),
            Arc::new(Activation::LeakyReLU(0.1)),
        ];
        let mut net = Network::new(&architecture, &activations, Metric::MSE, SGD { lr: 0.001 });

        // Input is a batch of 2 2D input vectors
        let input = arr2(&[[1.0, 2.0], [3.0, 4.5]]);
        let targets = arr2(&[[2.0, 9.8]]);

        // Capture the initial output before training
        let initial_output = net.forward(&input);
        println!("inital output: {initial_output}");

        // Train the network
        net.train(&input, &targets);

        // Capture the output after training
        let trained_output = net.forward(&input);
        println!("after training: {trained_output}");

        // Example assertion: check if the trained output is closer to the targets than the initial output
        // This requires calculating the loss for both and comparing them
        let initial_loss = Metric::MSE.calculate_loss(&initial_output, &targets);
        let trained_loss = Metric::MSE.calculate_loss(&trained_output, &targets);
        println!("{initial_loss}, {trained_loss}");
        assert!(
            trained_loss < initial_loss,
            "Training should reduce the loss"
        );
    }
}
