use crate::activation::Activation;
use core::fmt::Debug;
use ndarray::Array2;
use rand::Rng;
use std::sync::Arc;
impl Debug for dyn Activation {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Activation")
    }
}

#[derive(Debug, Clone)]
pub struct Weights {
    pub data: Array2<f64>,
}
#[derive(Debug, Clone)]
pub struct Biases {
    pub data: Array2<f64>,
}

#[derive(Debug, Clone)]
pub struct DenseLayer {
    pub input_dim: usize,
    pub output_dim: usize,
    pub weights: Weights,
    pub biases: Biases,
    pub activation: Arc<dyn Activation>,
    pub input: Array2<f64>,
}

impl DenseLayer {
    pub fn new(input_dim: usize, output_dim: usize, activation: Arc<dyn Activation>) -> Self {
        let mut rng = rand::thread_rng();
        let weights = Weights {
            data: Array2::from_shape_fn((output_dim, input_dim), |_| rng.gen_range(-1.0..1.0)),
        };
        let biases = Biases {
            data: Array2::zeros((output_dim, 1)),
        };

        let input = Array2::zeros((1, input_dim));
        DenseLayer {
            input_dim,
            output_dim,
            weights,
            biases,
            activation,
            input,
        }
    }

    pub fn forward(&mut self, input: &Array2<f64>) -> Array2<f64> {
        let wx: Array2<f64> = self.weights.data.dot(input);
        let linear_output = &wx + &self.biases.data;
        // Store input to this layer
        self.input = input.clone();

        self.activation.activate(linear_output)
    }

    pub fn grad_layer(
        &self,
        output_gradient: &Array2<f64>,
    ) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
        // This function calculates the gradients of this layer
        // 1. Apply the derivative of the activation function to the output_gradient.
        // 2. Calculate gradient w.r.t weights using `grad_weights`.
        // 3. Calculate gradient w.r.t biases using `grad_biases`.
        // 4. Calculate gradient w.r.t input for backpropagation to previous layers using `grad_input`.
        // Return (weight_gradient, bias_gradient, input_gradient)
        let activation_gradient = self.activation.calculate_gradient(output_gradient);

        // Calculate gradient with respect to weights
        let weight_gradient = self.grad_weights(&activation_gradient);

        // Calculate gradient with respect to biases
        let bias_gradient = self.grad_biases(&activation_gradient);

        // Calculate gradient with respect to input for backpropagation through previous layers
        let input_gradient = self.grad_input(&activation_gradient);

        // Instead of returning the weight and bias gradients, could store them on the Layer instead
        (weight_gradient, bias_gradient, input_gradient)
    }

    // Compute gradient of the loss with respect to weights
    pub fn grad_weights(&self, activation_gradient: &Array2<f64>) -> Array2<f64> {
        let grad_weights_updates = activation_gradient.dot(&self.input.t());
        grad_weights_updates.to_owned()
    }

    // Compute gradient of the loss with respect to biases
    pub fn grad_biases(&self, activation_gradient: &Array2<f64>) -> Array2<f64> {
        activation_gradient.to_owned() //.sum_axis(Axis(0)).insert_axis(Axis(0))
    }

    pub fn grad_input(&self, activation_gradient: &Array2<f64>) -> Array2<f64> {
        let input_grad = self.weights.data.t().dot(activation_gradient);
        input_grad
    }
}

#[cfg(test)]
mod test_layer {
    use super::*;
    use crate::activation::{ReLU, Sigmoid};
    use approx::assert_abs_diff_eq;
    use ndarray::arr2;
    use ndarray::Zip;

    fn arrays_are_close(a: &Array2<f64>, b: &Array2<f64>, tolerance: f64) -> bool {
        if a.shape() != b.shape() {
            return false;
        }

        Zip::from(a)
            .and(b)
            .fold(true, |acc, &a, &b| acc && (a - b).abs() <= tolerance)
    }

    #[test]
    fn test_dense_layer_forward() {
        // ReLU
        let mut layer = DenseLayer::new(2, 2, Arc::new(ReLU {}));
        layer.weights = Weights {
            data: arr2(&[[0.5, -0.5], [0.5, -0.5]]),
        };
        layer.biases = Biases {
            data: arr2(&[[2.1], [-5.0]]),
        };

        let input = arr2(&[[3.0], [2.0]]);
        let output = layer.forward(&input);
        let expected_output = arr2(&[[2.6], [0.0]]);

        assert_eq!(output, expected_output);

        let mut layer = DenseLayer::new(2, 2, Arc::new(ReLU {}));
        layer.weights = Weights {
            data: arr2(&[[0.5, -0.5], [0.5, -0.5]]),
        };
        layer.biases = Biases {
            data: arr2(&[[0.1], [0.2]]), // A (2 x 1) array
        };

        let input = arr2(&[[3.0], [2.0]]);
        let output = layer.forward(&input);

        let expected_output = arr2(&[[0.6], [0.7]]);

        assert_eq!(output, expected_output);

        // Sigmoid
        let mut layer = DenseLayer::new(2, 2, Arc::new(Sigmoid {}));
        layer.weights = Weights {
            data: arr2(&[[0.5, -0.5], [0.5, -0.5]]),
        };
        layer.biases = Biases {
            data: arr2(&[[0.1], [0.2]]),
        };

        let input = arr2(&[[3.0], [2.0]]);
        let output = layer.forward(&input);

        let expected_output = arr2(&[[0.6456563062257954], [0.6681877721681662]]);

        assert_eq!(output, expected_output);
    }

    #[test]
    fn test_grad_layer_2_by_2() {
        // Test 1
        let mut layer = DenseLayer::new(2, 2, Arc::new(ReLU {}));
        layer.weights = Weights {
            data: arr2(&[[0.5, 0.0], [0.5, 0.0]]),
        };
        layer.biases = Biases {
            data: arr2(&[[0.1], [0.2]]),
        };

        // Define a test input and a mock output gradient (as if coming from the next layer)
        let input = arr2(&[[1.0], [-1.0]]);
        let output = layer.forward(&input);
        let mock_input_grad_of_next_layer = arr2(&[[1.0], [1.0]]);
        let (weight_gradient, bias_gradient, input_gradient) =
            layer.grad_layer(&mock_input_grad_of_next_layer);
        let expected_weight_gradient = arr2(&[[1.0, -1.0], [1.0, -1.0]]); // FIXME - Update when implemented
        let expected_bias_gradient = arr2(&[[1.0], [1.0]]); // FIXME - Update when implemented
        let expected_input_gradient = arr2(&[[1.0], [0.0]]);

        let weight_gradient_diffs = expected_weight_gradient - weight_gradient;
        let bias_gradient_diffs = expected_bias_gradient - bias_gradient;
        let input_gradient_diffs = expected_input_gradient - input_gradient;

        for x in weight_gradient_diffs {
            assert_abs_diff_eq!(x, 0.0, epsilon = 1e-6);
        }
        for x in bias_gradient_diffs {
            assert_abs_diff_eq!(x, 0.0, epsilon = 1e-6);
        }
        for x in input_gradient_diffs {
            assert_abs_diff_eq!(x, 0.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_grad_layer_3_by_2() {
        // Test 2
        let mut layer = DenseLayer::new(2, 3, Arc::new(ReLU {}));
        layer.weights = Weights {
            data: arr2(&[[0.5, 0.0], [1.1, 0.5], [0.0, 2.3]]),
        };
        layer.biases = Biases {
            data: arr2(&[[0.1], [0.2], [1.1]]),
        };

        // Define a test input and a mock output gradient (as if coming from the next layer)
        let input = arr2(&[[1.0], [-1.0]]);
        let output = layer.forward(&input);
        let mock_input_grad_of_next_layer = arr2(&[[1.0], [1.0], [2.2]]);
        let (weight_gradient, bias_gradient, input_gradient) =
            layer.grad_layer(&mock_input_grad_of_next_layer);

        let expected_weight_gradient = arr2(&[[1.0, -1.0], [1.0, -1.0], [1.0, -1.0]]); // FIXME - Update when implemented
        let expected_bias_gradient = arr2(&[[1.0], [1.0], [1.0]]); // FIXME - Update when implemented
        let expected_input_gradient = arr2(&[[1.6], [2.8]]);

        let weight_gradient_diffs = expected_weight_gradient - weight_gradient;
        let bias_gradient_diffs = expected_bias_gradient - bias_gradient;
        let input_gradient_diffs = expected_input_gradient - input_gradient;

        for x in weight_gradient_diffs {
            assert_abs_diff_eq!(x, 0.0, epsilon = 1e-6);
        }
        for x in bias_gradient_diffs {
            assert_abs_diff_eq!(x, 0.0, epsilon = 1e-6);
        }
        for x in input_gradient_diffs {
            assert_abs_diff_eq!(x, 0.0, epsilon = 1e-6);
        }
    }
    #[test]
    fn test_batched_inputs_forward() {
        let mut layer = DenseLayer::new(2, 3, Arc::new(ReLU {}));
        layer.weights = Weights {
            data: arr2(&[[2.0, 0.0], [0.0, 2.0], [0.0, 1.0]]),
        };
        layer.biases = Biases {
            data: arr2(&[[0.1], [0.2], [0.3]]),
        };
        let input = arr2(&[[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]]);
        let output = layer.forward(&input);
        let expected_output = arr2(&[[2.3, 4.5, 6.7], [9.0, 11.2, 13.4], [4.7, 5.8, 6.9]]);
        assert!(arrays_are_close(&output, &expected_output, 0.00001));
    }
}
