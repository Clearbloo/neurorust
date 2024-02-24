use ndarray::{
    Array2,
    ArrayD,
    Axis,
    Ix2,
};
use rand::Rng;

pub struct DenseLayer {
    pub inputs: usize,
    pub outputs: usize,
    pub weights: Array2<f64>,
    pub biases: Array2<f64>,
    pub activation: ActivationFunction,
}

pub enum ActivationFunction {
    ReLU,
    LeakyReLU,
    Sigmoid,
}

impl DenseLayer {
    pub fn new(inputs: usize, outputs: usize, activation: ActivationFunction,) -> DenseLayer {
        let mut rng = rand::thread_rng();
        let weights = Array2::from_shape_fn((inputs, outputs), |_| rng.gen_range(-1.0..1.0));
        let biases = Array2::zeros((1, outputs));
        DenseLayer {
            inputs,
            outputs,
            weights,
            biases,
            activation,
        }
    }

    pub fn forward(&self, inputs: ArrayD<f64>) -> ArrayD<f64> {
        let inputs_mat = inputs.to_owned().into_dimensionality::<Ix2>().expect("Input must be 2D");
        let result = inputs_mat.dot(&self.weights) + &self.biases;
        self.activate(result.into_dyn())
    }

    pub fn activate(&self, input: ArrayD<f64>) -> ArrayD<f64> {
        match self.activation {
            ActivationFunction::ReLU => {
                input.mapv(|x| if x > 0.0 { x } else { 0.0 })
            },
            ActivationFunction::LeakyReLU => {
                input.mapv(|x| if x > 0.0 { x } else { 0.1 * x })
            },
            ActivationFunction::Sigmoid => {
                input.mapv(|x| 1.0 / (1.0 + (-x).exp()))
            },
        }
    }

    pub fn backprop(&self, inputs: &ArrayD<f64>, output_gradient: &ArrayD<f64>) -> (Array2<f64>, Array2<f64>) {
        let inputs_2d = inputs.view().into_dimensionality::<Ix2>().expect("Inputs must be 2D for dot product");
        let activation_gradient_2d = self.calculate_activation_gradient(output_gradient)
            .into_dimensionality::<Ix2>()
            .expect("Activation gradient must be 2D for dot product");

        // Calculate gradient with respect to weights
        let inputs_transposed = inputs_2d.t().to_owned();
        let weight_gradient = inputs_transposed.dot(&activation_gradient_2d);

        // Calculate gradient with respect to biases
        let bias_gradient = activation_gradient_2d.sum_axis(Axis(0)).insert_axis(Axis(0));

        (weight_gradient, bias_gradient)
    }

    fn calculate_activation_gradient(&self, output_gradient: &ArrayD<f64>) -> ArrayD<f64> {
        match self.activation {
            ActivationFunction::ReLU => {
                output_gradient.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 })
            },
            ActivationFunction::LeakyReLU => {
                output_gradient.mapv(|x| if x > 0.0 { 1.0 } else { 0.1 })
            },
            ActivationFunction::Sigmoid => {
                output_gradient.mapv(|x| x * (1.0 - x))
            },
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_dense_layer_forward() {
        let layer = DenseLayer {
            inputs: 2,
            outputs: 2,
            weights: arr2(&[[0.5, -0.5], [0.5, -0.5]]),
            biases: arr2(&[[0.1, 0.2]]),
            activation: ActivationFunction::ReLU
        };

        let input = arr2(&[[2.0, 3.0]]).into_dyn();
        let output = layer.forward(input);
        let expected_output = arr2(&[[2.6, 0.0]]).into_dyn();

        assert_eq!(output, expected_output);
    }

    #[test]
    fn test_dense_layer_forward_with_relu_activation() {
        let layer = DenseLayer {
            inputs: 2,
            outputs: 2,
            weights: arr2(&[[0.5, -0.5], [0.5, -0.5]]),
            biases: arr2(&[[0.1, 0.2]]),
            activation: ActivationFunction::ReLU,
        };

        let input = arr2(&[[2.0, 3.0]]).into_dyn();
        let output = layer.forward(input);

        // -0.8 becomes 0 due to ReLU
        let expected_output = arr2(&[[2.6, 0.0]]).into_dyn(); 

        assert_eq!(output, expected_output);
    }

    #[test]
    fn test_dense_layer_forward_with_sigmoid_activation() {
        let layer = DenseLayer {
            inputs: 2,
            outputs: 2,
            weights: arr2(&[[0.5, -0.5], [0.5, -0.5]]),
            biases: arr2(&[[0.1, 0.2]]),
            activation: ActivationFunction::Sigmoid,
        };

        let input = arr2(&[[2.0, 3.0]]).into_dyn();
        let output = layer.forward(input);
        
        let expected_output = arr2(&[[0.9308615796566533, 0.09112296101485616]]).into_dyn();

        assert_eq!(output, expected_output);
    }

    #[test]
    fn test_backprop() {
        let layer = DenseLayer {
            inputs: 2,
            outputs: 2,
            weights: arr2(&[[0.5, -0.5], [0.5, -0.5]]),
            biases: arr2(&[[0.1, 0.2]]),
            activation: ActivationFunction::LeakyReLU,
        };

        // Define a test input and a mock output gradient (as if coming from the next layer)
        let input = arr2(&[[2.0, 3.0]]).into_dyn();
        let output_gradient = arr2(&[[1.0, -1.0]]).into_dyn();

        let (weight_gradient, bias_gradient) = layer.backprop(&input, &output_gradient.into_dyn());

        let expected_weight_gradient = arr2(&[[2.0, 0.2], [3.0, 0.3]]);
        let expected_bias_gradient = arr2(&[[1.0, 0.1]]);

        let weight_gradient_diffs = expected_weight_gradient - weight_gradient;
        let bias_gradient_diffs = expected_bias_gradient - bias_gradient;

        for x in weight_gradient_diffs { assert_abs_diff_eq!(x, 0.0, epsilon = 1e-6);}
        for x in bias_gradient_diffs { assert_abs_diff_eq!(x, 0.0, epsilon = 1e-6);}
    }
}

