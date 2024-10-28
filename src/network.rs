use crate::{
    activation::Activation,
    layer::{Dense, LayerInitType},
    loss::Metric,
    optimizer::Optimizer,
};
use log::{debug, info};
use ndarray::Array2;

#[derive(Clone)]
pub struct Network {
    layers: Vec<Dense>,
    loss: Metric,
    optimizer: Optimizer,
}

impl Network {
    pub fn new(
        architecture: &[usize],
        activations: &[Activation],
        loss: Metric,
        optimizer: Optimizer,
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
                    LayerInitType::He,
                ));
            }
        }

        Self {
            layers,
            loss,
            optimizer,
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
        weight_updates.reverse();
        bias_updates.reverse();

        (weight_updates, bias_updates)
    }

    /// Gets the input for a specific layer.
    /// This is a placeholder for however you decide to implement it.
    pub fn get_input_for_layer(&self, layer_index: usize) -> Array2<f64> {
        // Return the input Array2<f64> for the specified layer.
        self.layers[layer_index].input.clone()
    }

    /// Each loop in epoch, forward pass, calculate loss, backwards pass to calculate gradients
    /// Update parameters (using optimizer), repeat.
    pub fn train(&mut self, epochs: usize, input: &Array2<f64>, targets: &Array2<f64>) {
        info!("Targets: {targets}");
        info!("Inputs: {input}");
        for e in 0..epochs {
            let outputs = self.forward(input);
            let (weight_updates, bias_updates) = self.backwards(&outputs, targets);
            let loss = self.loss.calculate_loss(&outputs, targets);
            if loss.is_nan() {
                panic!("found NAN in loss: {}", loss);
            }
            debug!("Epoch {e}");
            debug!("Layers:\n{:?}", self.layers);
            debug!("Outputs: {outputs}");
            debug!("Loss: {}", loss);
            debug!("Updates: {weight_updates:?}, {bias_updates:?}");
            if e % 10 == 0 {
                info!("Epoch {e}");
                info!("Loss: {}", self.loss.calculate_loss(&outputs, targets));
                info!("Outputs: {outputs}");
            }
            self.optimizer
                .apply_updates(&mut self.layers, &weight_updates, &bias_updates);
        }
    }

    pub fn get_params(&self) -> Vec<Array2<f64>> {
        let mut weights: Vec<Array2<f64>> = vec![];
        for layer in &self.layers {
            let mut w = vec![layer.weights.data.clone()];
            weights.append(&mut w);
        }
        weights
    }

    pub fn load_params(self, params: Vec<Array2<f64>>) {
        for (i, mut layer) in self.layers.into_iter().enumerate() {
            layer.weights.data = params[i].clone()
        }
    }
}

#[cfg(test)]
mod test_network {
    use super::*;
    use crate::{activation::Activation, loss::Metric, utils::min_max_scale};
    use ctor::ctor;
    use log::debug;
    use ndarray::arr2;

    #[ctor]
    fn init_logger() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    #[test]
    fn test_init() {
        let architecture_1 = vec![1, 2, 1];
        let activations: Vec<Activation> = vec![Activation::ReLU, Activation::LeakyReLU(0.1)];
        let net1 = Network::new(
            &architecture_1,
            &activations,
            Metric::MSE,
            Optimizer::Adam { lr: 0.001 },
        );

        assert_eq!(net1.layers.len(), 2);

        let architecture_2 = vec![1, 1];
        let activations_2: Vec<Activation> = vec![Activation::ReLU];
        let net2 = Network::new(
            &architecture_2,
            &activations_2,
            Metric::MSE,
            Optimizer::Adam { lr: 0.001 },
        );

        assert_eq!(net2.layers.len(), 1);
    }

    #[test]
    fn test_train_updates_weights() {
        // This test should be usurped by the below eventually
        let architecture = vec![1, 2, 1];
        let activations: Vec<Activation> =
            vec![Activation::LeakyReLU(0.1), Activation::LeakyReLU(0.1)];
        let mut net = Network::new(
            &architecture,
            &activations,
            Metric::MSE,
            Optimizer::GD { lr: 0.1 },
        );

        let input = arr2(&[[1.0]]);
        let targets = arr2(&[[2.0]]);
        let y_hat1 = net.forward(&input);

        net.train(50, &input, &targets);
        let y_hat2 = net.forward(&input);
        assert_ne!(y_hat1, y_hat2);
    }

    #[test]
    fn test_train_reduces_loss_simple() {
        // Test 1 - simple input output
        // Should merge this with the one below
        let architecture = vec![1, 1];
        let activations: Vec<Activation> = vec![Activation::LeakyReLU(0.1)];
        let mut net = Network::new(
            &architecture,
            &activations,
            Metric::MSE,
            Optimizer::GD { lr: 0.01 },
        );

        let input = arr2(&[[1.0]]);
        let targets = arr2(&[[2.0]]);
        let y1 = net.forward(&input);

        net.train(1000, &input, &targets);
        let y2 = net.forward(&input);
        assert_ne!(y1, y2);

        let initial_loss = Metric::MSE.calculate_loss(&y1, &targets);
        let trained_loss = Metric::MSE.calculate_loss(&y2, &targets);
        debug!("Initial loss: {initial_loss}, Trained_loss: {trained_loss}");
        assert!(
            trained_loss < initial_loss,
            "Training should reduce the loss"
        );
        assert!(trained_loss < 1e-6, "Loss should be near 0");
    }

    #[test]
    fn test_train_reduces_loss_2() {
        // Test 2 - Hidden layer
        let architecture = vec![2, 4, 1];
        let activations: Vec<Activation> =
            vec![Activation::LeakyReLU(0.9), Activation::LeakyReLU(0.9)];
        let mut net = Network::new(
            &architecture,
            &activations,
            Metric::MSE,
            Optimizer::GD { lr: 0.1 },
        );

        // Input is a batch of 2 2D input vectors
        let input = min_max_scale(&arr2(&[[1.0, 2.0, 5.0], [3.0, 4.5, 6.0]]));
        let targets = min_max_scale(&arr2(&[[2.0, -1.8, 4.0]]));

        // Test the shapes of the output
        let y1 = net.forward(&arr2(&[[1.0], [3.0]]));
        assert_eq!(y1.shape(), [1, 1], "One input should give one output");

        // Capture the initial output before training
        let y2 = net.forward(&input);
        assert_eq!(y2.shape(), [1, 3], "Three inputs gives three outputs");

        // Train the network
        net.train(1000, &input, &targets);

        // Capture the output after training
        let trained_output = net.forward(&input);
        info!("output after training: {trained_output}");

        // Example assertion: check if the trained output is closer to the targets than the initial output
        // This requires calculating the loss for both and comparing them
        let initial_loss = Metric::MSE.calculate_loss(&y2, &targets);
        let trained_loss = Metric::MSE.calculate_loss(&trained_output, &targets);
        debug!("Initial loss: {initial_loss}, Trained_loss: {trained_loss}");
        assert!(
            trained_loss < initial_loss,
            "Training should reduce the loss"
        );
        assert!(trained_loss < 1e-1, "Loss should be near 0: {trained_loss}");
    }
}
