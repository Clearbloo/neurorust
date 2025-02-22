use crate::activation::Activation;
use core::fmt::Debug;
use ndarray::{Array2, Axis};
use rand_distr::{Distribution, Normal, Uniform};

type Matrix = Array2<f64>;

#[derive(Clone)]
pub struct Weights {
    pub data: Matrix,
}
#[derive(Clone)]
pub struct Biases {
    pub data: Matrix,
}

// TODO - rename input to x and put input as a comment
#[derive(Debug, Clone)]
pub struct Dense {
    pub input_dim: usize,
    pub output_dim: usize,
    pub weights: Weights,
    pub biases: Biases,
    pub activation: Activation,
    pub input: Matrix,
    pub z: Matrix, // pre-activation output
}

pub enum InitType {
    Uniform,
    He,
    Xavier,
}

impl Debug for Weights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Weights")
            .field(
                "data",
                &self.data.iter().collect::<Vec<_>>(), // Collect elements into a Vec for clean output
            )
            .finish()
    }
}

impl Debug for Biases {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Biases")
            .field(
                "data",
                &self.data.iter().collect::<Vec<_>>(), // Collect elements into a Vec for clean output
            )
            .finish()
    }
}
/// TODO - implement a trait so that we can make other kinds of layers
impl Dense {
    /// # Panics
    /// Panics if the standard deviation is not finite
    #[must_use]
    pub fn new(
        input_dim: usize,
        output_dim: usize,
        activation: Activation,
        init_type: InitType,
    ) -> Self {
        let mut rng = rand::thread_rng();
        let weights = match init_type {
            InitType::Uniform => todo!(),
            InitType::Xavier => {
                let limit = (6.0 / (input_dim + output_dim) as f64).sqrt();
                let uniform = Uniform::new(-limit, limit);
                Weights {
                    data: Array2::from_shape_fn((output_dim, input_dim), |_| {
                        uniform.sample(&mut rng)
                    }),
                }
            }
            InitType::He => {
                let stddev = (2.0 / input_dim as f64).sqrt(); // He initialization
                let normal = Normal::new(0.0, stddev).unwrap();
                Weights {
                    data: Array2::from_shape_fn((output_dim, input_dim), |_| {
                        normal.sample(&mut rng)
                    }),
                }
            }
        };
        let biases = Biases {
            data: Array2::zeros((output_dim, 1)),
        };

        // TODO - Pass in batch size to init, rather than assume its 1
        let input = Array2::zeros((input_dim, 1));
        let z = Array2::zeros((output_dim, 1));
        Self {
            input_dim,
            output_dim,
            weights,
            biases,
            activation,
            input,
            z,
        }
    }

    pub fn forward(&mut self, input: &Matrix) -> Matrix {
        let wx: Matrix = self.weights.data.dot(input);
        // Uses broadcasting rules for addition
        let linear_output = &wx + &self.biases.data;

        self.input = input.clone();
        self.z = linear_output.clone();

        self.activation.activate(&linear_output)
    }

    pub fn predict(&self, input: &Matrix) -> Matrix {
        let wx: Matrix = self.weights.data.dot(input);
        // Uses broadcasting rules for addition
        let linear_output = &wx + &self.biases.data;
        self.activation.activate(&linear_output)
    }

    /// This function calculates the gradients of this layer
    /// 1. Apply the derivative of the activation function to the `output_gradient`.
    /// 2. Calculate gradient w.r.t weights using `grad_weights`.
    /// 3. Calculate gradient w.r.t biases using `grad_biases`.
    /// 4. Calculate gradient w.r.t input for backpropagation to previous layers using `grad_input`.
    /// 5. Return (`weight_gradient`, `bias_gradient`, `input_gradient`)
    ///
    /// TODO - rename this to just grad
    #[must_use]
    pub fn grad_layer(&self, output_gradient: &Matrix) -> (Matrix, Matrix, Matrix) {
        // Compute activation derivative
        let activation_derivative = self.activation.calculate_derivative(&self.z);
        // Element-wise multiplication
        let delta = output_gradient * &activation_derivative;

        // Calculate gradients
        let weight_gradient = self.grad_weights(&delta);
        let bias_gradient = self.grad_biases(&delta);
        let input_gradient = self.grad_input(&delta);

        // Instead of returning the weight and bias gradients, could store them on the Layer instead
        (weight_gradient, bias_gradient, input_gradient)
    }

    /// Compute gradient of the loss with respect to weights
    #[must_use]
    pub fn grad_weights(&self, activation_gradient: &Matrix) -> Matrix {
        let grad_weights_updates = activation_gradient.dot(&self.input.t());
        grad_weights_updates.to_owned()
    }

    /// Compute gradient of the loss with respect to biases
    #[must_use]
    pub fn grad_biases(&self, activation_gradient: &Matrix) -> Matrix {
        activation_gradient
            .sum_axis(Axis(1))
            .insert_axis(Axis(1))
            .to_owned()
    }

    #[must_use]
    pub fn grad_input(&self, activation_gradient: &Matrix) -> Matrix {
        let input_grad = self.weights.data.t().dot(activation_gradient);
        input_grad
    }
}

#[cfg(test)]
mod test_layer {
    use super::*;
    use crate::utils::arrays_are_close;
    use approx::assert_abs_diff_eq;
    use ctor::ctor;
    use log::debug;
    use ndarray::arr2;

    #[ctor]
    fn init_logger() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    #[test]
    fn test_dense_layer_forward() {
        // Test 1 - Leaky ReLU
        let mut layer = Dense::new(2, 2, Activation::LeakyReLU(0.1), InitType::He);
        layer.weights = Weights {
            data: arr2(&[[0.5, -0.5], [0.5, -0.5]]),
        };
        layer.biases = Biases {
            data: arr2(&[[2.1], [-5.0]]),
        };

        let input = arr2(&[[3.0], [2.0]]);
        let output = layer.forward(&input);
        let expected_output = arr2(&[[2.6], [-0.45]]);
        assert_eq!(output, expected_output);

        // Test 2 - ReLU
        let mut layer = Dense::new(2, 2, Activation::ReLU, InitType::He);
        layer.weights = Weights {
            data: arr2(&[[0.5, -0.5], [0.5, -0.5]]),
        };
        layer.biases = Biases {
            data: arr2(&[[0.1], [-2.2]]),
        };

        let input = arr2(&[[3.0], [2.0]]);
        let output = layer.forward(&input);
        let expected_output = arr2(&[[0.6], [0.0]]);
        assert_eq!(output, expected_output);

        // Test 3 - Sigmoid
        let mut layer = Dense::new(2, 2, Activation::Sigmoid, InitType::He);
        layer.weights = Weights {
            data: arr2(&[[0.5, -0.5], [0.5, -0.5]]),
        };
        layer.biases = Biases {
            data: arr2(&[[0.1], [0.2]]),
        };

        let input = arr2(&[[3.0], [2.0]]);
        let output = layer.forward(&input);
        let expected_output = arr2(&[[0.645_656_306_225_795_4], [0.668_187_772_168_166_2]]);
        assert_eq!(output, expected_output);
    }

    #[test]
    fn test_grad_layer_2_by_2() {
        // Test 1
        let mut layer = Dense::new(2, 2, Activation::ReLU, InitType::He);
        layer.weights = Weights {
            data: arr2(&[[0.5, 0.0], [0.5, 0.0]]),
        };
        layer.biases = Biases {
            data: arr2(&[[0.1], [0.2]]),
        };

        // Define a test input and a mock output gradient,
        // as if coming from the next layer

        // Set the layer input with a forward pass
        let input = arr2(&[[1.0], [-1.0]]);
        layer.forward(&input);

        let mock_input_grad_of_next_layer = arr2(&[[1.0], [1.0]]);
        let (weight_gradient, bias_gradient, input_gradient) =
            layer.grad_layer(&mock_input_grad_of_next_layer);
        let expected_weight_gradient = arr2(&[[1.0, -1.0], [1.0, -1.0]]);
        let expected_bias_gradient = arr2(&[[1.0], [1.0]]);
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
        let mut layer = Dense::new(2, 3, Activation::LeakyReLU(0.1), InitType::He);
        layer.weights = Weights {
            data: arr2(&[[0.5, 0.0], [1.1, 0.5], [0.0, 2.3]]),
        };
        layer.biases = Biases {
            data: arr2(&[[0.1], [0.2], [1.1]]),
        };

        // Define a test input and a mock output gradient (as if coming from the next layer)

        // We need to make a forward pass to set the input
        let input = arr2(&[
            [1.0, 2.5, 3.6, -4.4, 100.1],
            [1.0, -1.4, -7.7, 345.9, 239_487.123_234],
        ]);
        layer.forward(&input);
        let mock_input_grad_of_next_layer = arr2(&[
            [1.0, 2.0, 3.0, 1.0, 4.4],
            [5.5, 2.2, 6.6, 23.3, 23.65],
            [4.4, 234.5, 4.3, 7.8, 3.2],
        ]);
        let (weight_gradient, bias_gradient, input_gradient) =
            layer.grad_layer(&mock_input_grad_of_next_layer);

        let expected_weight_gradient = arr2(&[
            [456.8, 1_053_753.032_229_6],
            [2_299.604_999_999_999_6, 5_671_881.534_484_1],
            [350.573, 769_025.073_348_800_1],
        ]);
        let expected_bias_gradient = arr2(&[[10.5], [61.25], [39.28]]);
        let expected_input_gradient = arr2(&[
            [
                6.550_000_000_000_001,
                3.420_000_000_000_000_4,
                8.76,
                25.680_000_000_000_003,
                28.215,
            ],
            [12.87, 55.035_000_000_000_004, 4.289, 29.59, 19.185],
        ]);
        debug!("Wgrad:\n{weight_gradient},\n Bgrad:\n{bias_gradient},\n Igrad:\n{input_gradient}");

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
        let mut layer = Dense::new(2, 3, Activation::ReLU, InitType::He);
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
    #[test]
    fn test_zero_loss() {
        // If there's no loss, there should be no gradient.
        let mut layer = Dense::new(2, 3, Activation::ReLU, InitType::He);
        let input = Array2::from_elem([2, 88], 0.0);
        layer.forward(&input);

        let loss: Matrix = Array2::from_elem([3, 88], 0.0);
        let (w, b, i) = layer.grad_layer(&loss);
        for x in &w {
            assert!(x == &0.0);
        }
        for x in &b {
            assert!(x == &0.0);
        }
        for x in &i {
            assert!(x == &0.0);
        }
    }
}
