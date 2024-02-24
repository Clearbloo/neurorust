use crate::activation::Activation;
use crate::gradient::Grad;
use ndarray::{Array2, Ix2};
use rand::Rng;

pub struct Weights {
    data: Array2<f64>,
}
impl Grad for Weights {
    fn grad(&self, output: &Array2<f64>) -> Array2<f64> {
        // TODO - implement weight gradient
        output.clone()
    }
}

pub struct Biases {
    data: Array2<f64>,
}
impl Grad for Biases {
    fn grad(&self, output: &Array2<f64>) -> Array2<f64> {
        // TODO - implement bias gradient
        output.clone()
    }
}

pub struct DenseLayer<A: Activation> {
    pub input_dim: usize,
    pub output_dim: usize,
    pub weights: Weights,
    pub biases: Biases,
    pub activation: A,
}

impl<A: Activation> DenseLayer<A> {
    pub fn new(input_dim: usize, output_dim: usize, activation: A) -> Self {
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
        }
    }
}

// impl<A> Grad for DenseLayer<A> where A: Activation{
//     fn grad(&self, output: &Array2<f64>) -> Array2<f64> {
//         self.activation.grad(output);
//         self.weights.grad(output);
//         self.biases.grad(output)
//     }
// }

impl<A: Activation> DenseLayer<A> {
    pub fn forward(&self, inputs: Array2<f64>) -> Array2<f64> {
        let inputs_mat = inputs
            .to_owned()
            .into_dimensionality::<Ix2>()
            .expect("Input must be 2D");
        let result = inputs_mat.dot(&self.weights.data) + &self.biases.data;
        self.activation.activate(result)
    }

    pub fn backprop(&self, output: &Array2<f64>) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
        let activation_gradient = self.activation.grad(output);
        // .expect("Activation gradient must be 2D for dot product");

        // Calculate gradient with respect to weights
        let weight_gradient = self.weights.grad(output);
        // Calculate gradient with respect to biases
        let bias_gradient = self.biases.grad(output);

        (activation_gradient, weight_gradient, bias_gradient)
    }
}

#[cfg(test)]
mod test_layer {
    use super::*;
    use crate::activation::{LeakyReLU, ReLU, Sigmoid};
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
        };

        // Define a test input and a mock output gradient (as if coming from the next layer)
        let output = arr2(&[[1.0, -1.0], [2.0, 3.0]]);

        let (activation_gradient, weight_gradient, bias_gradient) = layer.backprop(&output);

        let expected_activation_gradient = arr2(&[[1.0, 0.1], [1.0, 1.0]]);
        let expected_weight_gradient = arr2(&[[1.0, -1.0], [2.0, 3.0]]); // FIXME - Update when implemented
        let expected_bias_gradient = arr2(&[[1.0, -1.0], [2.0, 3.0]]); // FIXME - Update when implemented

        println!("Weight: {}, Bias: {}, Activation: {}", weight_gradient, bias_gradient, activation_gradient);
        
        let activation_gradient_diffs = expected_activation_gradient - activation_gradient;
        let weight_gradient_diffs = expected_weight_gradient - weight_gradient;
        let bias_gradient_diffs = expected_bias_gradient - bias_gradient;


        for x in activation_gradient_diffs {
            assert_abs_diff_eq!(x, 0.0, epsilon = 1e-6);
        }
        for x in weight_gradient_diffs {
            assert_abs_diff_eq!(x, 0.0, epsilon = 1e-6);
        }
        for x in bias_gradient_diffs {
            assert_abs_diff_eq!(x, 0.0, epsilon = 1e-6);
        }
    }
}
