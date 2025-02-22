use crate::utils::Matrix;
use core::fmt::Debug;

#[derive(Debug, Clone)]
pub enum Activation {
    Linear,
    ReLU,
    Sigmoid,
    LeakyReLU(f64),
}

impl Activation {
    #[must_use]
    pub fn activate(&self, x: &Matrix) -> Matrix {
        match self {
            Self::Linear => x.clone(),
            Self::ReLU => relu(x),
            Self::LeakyReLU(slope) => leaky_relu(x, *slope),
            Self::Sigmoid => sigmoid(x),
        }
    }
    /// Returns the gradient of the activation function. Same shape as the
    /// `output_gradient` parameter
    #[must_use]
    pub fn calculate_derivative(&self, pre_activation_output: &Matrix) -> Matrix {
        match self {
            Self::Linear => {
                let x_dim = pre_activation_output.shape()[0];
                let y_dim = pre_activation_output.shape()[1];
                Matrix::from_elem((x_dim, y_dim), 1.)
            }
            Self::ReLU => relu_derivative(pre_activation_output),
            Self::LeakyReLU(slope) => leaky_relu_derivative(pre_activation_output, *slope),
            Self::Sigmoid => sigmoid_derivative(pre_activation_output),
        }
    }
}

// Activation functions
fn relu(input: &Matrix) -> Matrix {
    input.mapv(|x| if x > 0.0 { x } else { 0.0 })
}

fn leaky_relu(input: &Matrix, slope: f64) -> Matrix {
    input.mapv(|x| if x > 0.0 { x } else { slope * x })
}

fn sigmoid(input: &Matrix) -> Matrix {
    input.mapv(|x| 1.0 / (1.0 + (-x).exp()))
}

fn relu_derivative(relu_output: &Matrix) -> Matrix {
    relu_output.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 })
}

fn leaky_relu_derivative(leaky_relu_output: &Matrix, slope: f64) -> Matrix {
    leaky_relu_output.mapv(|x| if x > 0.0 { 1.0 } else { slope })
}

fn sigmoid_derivative(sigmoid_output: &Matrix) -> Matrix {
    sigmoid_output * &(1.0 - sigmoid_output)
}

#[cfg(test)]
mod test_activations {
    use super::Activation;
    use ndarray::arr2;

    #[test]
    fn test_relu() {
        let relu = Activation::ReLU;
        let input = arr2(&[[1.0, -2.0], [2.0, -3.0]]);
        let result = relu.activate(&input);
        assert_eq!(result, arr2(&[[1.0, 0.0], [2.0, 0.0]]));

        let act_grad = relu.calculate_derivative(&result);
        assert_eq!(act_grad, arr2(&[[1.0, 0.0], [1.0, 0.0]]));
    }
}
