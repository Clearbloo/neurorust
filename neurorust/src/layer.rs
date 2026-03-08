use crate::activation::Activation;
use crate::gradient::Grad;
use crate::stage::Predictor;
use crate::utils::Matrix;
use core::fmt::Debug;
use ndarray::Array2;
use rand_distr::{Distribution, Normal, Uniform};

#[derive(Debug, Clone)]
pub struct Weights;
#[derive(Debug, Clone)]
pub struct Firing;

#[derive(Debug, Clone)]
pub struct Linear {
    pub input_dim: usize,
    pub output_dim: usize,
    pub weights: Matrix<Weights>,
    pub activation: Activation,
}

pub enum InitType {
    Uniform,
    He,
    Xavier,
}

/// TODO - implement a trait so that we can make other kinds of layers
impl Linear {
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
                Weights(Array2::from_shape_fn((output_dim, input_dim), |_| {
                    uniform.sample(&mut rng)
                }))
            }
            InitType::He => {
                let stddev = (2.0 / input_dim as f64).sqrt(); // He initialization
                let normal = Normal::new(0.0, stddev).unwrap();
                Weights(Array2::from_shape_fn((output_dim, input_dim), |_| {
                    normal.sample(&mut rng)
                }))
            }
        };
        let biases = Bias(Array2::zeros((output_dim, 1)));

        // TODO - Pass in batch size to init, rather than assume its 1
        let input = Array2::zeros((input_dim, 1));
        let z = Array2::zeros((output_dim, 1));
        Self {
            input_dim,
            output_dim,
            weights,
            activation,
        }
    }
}

impl Linear {
    fn z(&self, input: &Matrix<Firing>) -> Matrix<Firing> {
        Matrix::<Firing>(self.weights.dot(input.as_ref()))
    }
    fn a(&self, z: &Matrix<Firing>) -> Matrix<Firing> {
        self.activation.activate(z)
    }
}

// TODO - Add option of a bias
impl Predictor for Linear {
    type Input = Matrix<Firing>;
    type Output = Matrix<Firing>;
    type Parameters = Matrix<Weights>;

    fn forward(&self, input: &Self::Input) -> Self::Output {
        self.a(&self.z(input))
    }

    fn parameters(&self) -> &Self::Parameters {
        &self.weights
    }
}

impl Grad for Linear {
    /// This function calculates the gradients of this layer
    /// 1. Apply the derivative of the activation function to the `output_gradient`.
    /// 2. Calculate gradient w.r.t weights using `grad_weights`.
    /// 3. Calculate gradient w.r.t biases using `grad_biases`.
    /// 4. Calculate gradient w.r.t input for backpropagation to previous layers using `grad_input`.
    /// 5. Return (`weight_gradient`, `bias_gradient`)
    #[must_use]
    fn grad(&self, data: Self::Input) -> (Self::Parameters, Self::Output) {
        let z = self.z(&data);
        let a = self.a(&z);

        // da_l/dW_l
        let da_dz = self.activation.calculate_derivative(&z);
        let dz_dW = data;
        let da_dW = &da_dz * dz_dW;

        // da_l/da_{l-1}
        let dz_da = &self.weights;
        let da_da = &da_dz * dz_da;

        // Instead of returning the weight and bias gradients, could store them on the Layer instead
        (da_dW, da_da)
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
        let mut layer = Linear::new(2, 2, Activation::LeakyReLU(0.1), InitType::He);
        layer.weights = Weights(arr2(&[[0.5, -0.5], [0.5, -0.5]]));
        layer.biases = Bias(arr2(&[[2.1], [-5.0]]));

        let input = arr2(&[[3.0], [2.0]]);
        let output = layer.forward(&input);
        let expected_output = arr2(&[[2.6], [-0.45]]);
        assert_eq!(output, expected_output);

        // Test 2 - ReLU
        let mut layer = Linear::new(2, 2, Activation::ReLU, InitType::He);
        layer.weights = Weights(arr2(&[[0.5, -0.5], [0.5, -0.5]]));
        layer.biases = Bias(arr2(&[[0.1], [-2.2]]));

        let input = arr2(&[[3.0], [2.0]]);
        let output = layer.forward(&input);
        let expected_output = arr2(&[[0.6], [0.0]]);
        assert_eq!(output, expected_output);

        // Test 3 - Sigmoid
        let mut layer = Linear::new(2, 2, Activation::Sigmoid, InitType::He);
        layer.weights = Weights(arr2(&[[0.5, -0.5], [0.5, -0.5]]));
        layer.biases = Bias(arr2(&[[0.1], [0.2]]));

        let input = arr2(&[[3.0], [2.0]]);
        let output = layer.forward(&input);
        let expected_output = arr2(&[[0.645_656_306_225_795_4], [0.668_187_772_168_166_2]]);
        assert_eq!(output, expected_output);
    }

    #[test]
    fn test_grad_layer_2_by_2() {
        // Test 1
        let mut layer = Linear::new(2, 2, Activation::ReLU, InitType::He);
        layer.weights = Weights(arr2(&[[0.5, 0.0], [0.5, 0.0]]));
        layer.biases = Bias(arr2(&[[0.1], [0.2]]));

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
        let mut layer = Linear::new(2, 3, Activation::LeakyReLU(0.1), InitType::He);
        layer.weights = Weights(arr2(&[[0.5, 0.0], [1.1, 0.5], [0.0, 2.3]]));
        layer.biases = Bias(arr2(&[[0.1], [0.2], [1.1]]));

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
        let mut layer = Linear::new(2, 3, Activation::ReLU, InitType::He);
        layer.weights = Weights(arr2(&[[2.0, 0.0], [0.0, 2.0], [0.0, 1.0]]));
        layer.biases = Bias(arr2(&[[0.1], [0.2], [0.3]]));
        let input = arr2(&[[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]]);
        let output = layer.forward(&input);
        let expected_output = arr2(&[[2.3, 4.5, 6.7], [9.0, 11.2, 13.4], [4.7, 5.8, 6.9]]);
        assert!(arrays_are_close(&output, &expected_output, 0.00001));
    }
    #[test]
    fn test_zero_loss() {
        // If there's no loss, there should be no gradient.
        let mut layer = Linear::new(2, 3, Activation::ReLU, InitType::He);
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
