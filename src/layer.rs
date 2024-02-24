use crate::{activation::Activation, loss::Loss};
use ndarray::{Array2, Axis};
use rand::Rng;

pub struct Weights {
    data: Array2<f64>,
}

pub struct Biases {
    data: Array2<f64>,
}

pub struct DenseLayer<A: Activation, L: Loss> {
    pub input_dim: usize,
    pub output_dim: usize,
    pub weights: Weights,
    pub biases: Biases,
    pub activation: A,
    pub loss: L,
}

impl<A: Activation, L: Loss> DenseLayer<A, L> {
    pub fn new(input_dim: usize, output_dim: usize, activation: A, loss: L) -> Self {
        let mut rng = rand::thread_rng();
        let weights = Weights {
            data: Array2::from_shape_fn((input_dim, output_dim), |_| rng.gen_range(-1.0..1.0)),
        };
        let biases = Biases {
            data: Array2::zeros((1, output_dim)),
        };
        DenseLayer {
            input_dim,
            output_dim,
            weights,
            biases,
            activation,
            loss,
        }
    }
}

impl<A: Activation, L: Loss> DenseLayer<A, L> {
    pub fn forward(&self, input: Array2<f64>) -> Array2<f64> {
        let linear_output = input.dot(&self.weights.data) + &self.biases.data;
        self.activation.activate(linear_output)
    }

    pub fn backprop(
        &self,
        input: &Array2<f64>,
        output: &Array2<f64>,
        target: &Array2<f64>,
    ) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
        // Calculate output gradient with respect to loss function
        let loss_gradient = self.calculate_output_gradient(output, target, &self.loss);

        // Apply the derivative of the activation function to the loss gradient
        // This requires calculating the pre-activated output (Z) again, which is not ideal. Consider storing Z during the forward pass if possible.
        let linear_output = input.dot(&self.weights.data) + &self.biases.data;
        let activation_gradient = self.activation.calculate_gradient(&linear_output);
        let output_gradient = loss_gradient * activation_gradient;

        // Calculate gradient with respect to weights
        let weight_gradient = self.grad_weights(input, &output_gradient);

        // Calculate gradient with respect to biases
        let bias_gradient = self.grad_biases(&output_gradient);

        // Optionally, calculate gradient with respect to input for backpropagation through previous layers
        let input_gradient = self.grad_input(&output_gradient);

        (weight_gradient, bias_gradient, input_gradient)
    }

    pub fn calculate_output_gradient(
        &self,
        predictions: &Array2<f64>,
        targets: &Array2<f64>,
        loss_function: &dyn Loss,
    ) -> Array2<f64> {
        loss_function.calculate_gradient(predictions, targets)
    }

    // Compute gradient of the loss with respect to weights
    pub fn grad_weights(&self, input: &Array2<f64>, output_gradient: &Array2<f64>) -> Array2<f64> {
        let input_transposed = input.t();
        input_transposed.dot(output_gradient)
    }

    // Compute gradient of the loss with respect to biases
    pub fn grad_biases(&self, output_gradient: &Array2<f64>) -> Array2<f64> {
        output_gradient.sum_axis(Axis(0)).insert_axis(Axis(0))
    }

    // Optionally, if you need to compute the gradient with respect to the input for backpropagation
    pub fn grad_input(&self, output_gradient: &Array2<f64>) -> Array2<f64> {
        output_gradient.dot(&self.weights.data.t())
    }
}

#[cfg(test)]
mod test_layer {
    use super::*;
    use crate::activation::{LeakyReLU, ReLU, Sigmoid};
    use crate::loss::MeanSquaredError;
    use approx::assert_abs_diff_eq;
    use ndarray::arr2;

    #[test]
    fn test_dense_layer_forward() {
        let layer = DenseLayer {
            input_dim: 2,
            output_dim: 2,
            weights: Weights {
                data: arr2(&[[0.5, -0.5], [0.5, -0.5]]),
            },
            biases: Biases {
                data: arr2(&[[0.1, 0.2]]),
            },
            activation: ReLU,
            loss: MeanSquaredError,
        };

        let input = arr2(&[[2.0, 3.0]]);
        let output = layer.forward(input);
        let expected_output = arr2(&[[2.6, 0.0]]);

        assert_eq!(output, expected_output);
    }

    #[test]
    fn test_dense_layer_forward_with_relu_activation() {
        let layer = DenseLayer {
            input_dim: 2,
            output_dim: 2,
            weights: Weights {
                data: arr2(&[[0.5, -0.5], [0.5, -0.5]]),
            },
            biases: Biases {
                data: arr2(&[[0.1, 0.2]]),
            },
            activation: ReLU,
            loss: MeanSquaredError,
        };

        // let layer = DenseLayer::new(2, 2, ReLU);

        let input = arr2(&[[2.0, 3.0]]);
        let output = layer.forward(input);

        // -0.8 becomes 0 due to ReLU
        let expected_output = arr2(&[[2.6, 0.0]]);

        assert_eq!(output, expected_output);
    }

    #[test]
    fn test_dense_layer_forward_with_sigmoid_activation() {
        let layer = DenseLayer {
            input_dim: 2,
            output_dim: 2,
            weights: Weights {
                data: arr2(&[[0.5, -0.5], [0.5, -0.5]]),
            },
            biases: Biases {
                data: arr2(&[[0.1, 0.2]]),
            },
            activation: Sigmoid,
            loss: MeanSquaredError,
        };

        let input = arr2(&[[2.0, 3.0]]);
        let output = layer.forward(input);

        let expected_output = arr2(&[[0.9308615796566533, 0.09112296101485616]]);

        assert_eq!(output, expected_output);
    }

    #[test]
    fn test_backprop() {
        let layer = DenseLayer {
            input_dim: 2,
            output_dim: 2,
            weights: Weights {
                data: arr2(&[[0.5, -0.5], [0.5, -0.5]]),
            },
            biases: Biases {
                data: arr2(&[[0.1, 0.2]]),
            },
            activation: LeakyReLU,
            loss: MeanSquaredError,
        };

        // Define a test input and a mock output gradient (as if coming from the next layer)
        let input = arr2(&[[1.0, -1.0], [2.0, 3.0]]);
        let output = layer.forward(input.clone());
        let targets = arr2(&[[2.0, 1.0]]);
        let (weight_gradient, bias_gradient, _input_gradient) =
            layer.backprop(&input, &output, &targets);

        let _expected_input_gradient = arr2(&[[1.0, 0.1], [1.0, 1.0]]);
        let expected_weight_gradient = arr2(&[[-0.7, -1.046], [3.7, 0.431]]); // FIXME - Update when implemented
        let expected_bias_gradient = arr2(&[[-1.3, -0.923]]); // FIXME - Update when implemented

        println!("Weight: {}, Bias: {}", weight_gradient, bias_gradient);

        let weight_gradient_diffs = expected_weight_gradient - weight_gradient;
        let bias_gradient_diffs = expected_bias_gradient - bias_gradient;

        for x in weight_gradient_diffs {
            assert_abs_diff_eq!(x, 0.0, epsilon = 1e-6);
        }
        for x in bias_gradient_diffs {
            assert_abs_diff_eq!(x, 0.0, epsilon = 1e-6);
        }
    }
}
