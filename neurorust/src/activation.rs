use crate::{layer::Firing, utils::Matrix};
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
    pub fn activate(&self, x: &Matrix<Firing>) -> Matrix<Firing> {
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
    pub fn calculate_derivative(&self, pre_activation_output: &Matrix<Firing>) -> Matrix<Firing> {
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
fn relu(mut input: Matrix<Firing>) -> Matrix<Firing> {
    input.0.mapv_inplace(|x| if x > 0.0 { x } else { 0.0 });
    input
}

fn leaky_relu(mut input: Matrix<Firing>, slope: f64) -> Matrix<Firing> {
    input
        .0
        .mapv_inplace(|x| if x > 0.0 { x } else { slope * x });
    input
}

fn sigmoid(mut input: Matrix<Firing>) -> Matrix<Firing> {
    input.0.mapv_inplace(|x| 1.0 / (1.0 + (-x).exp()));
    input
}

fn relu_derivative(mut relu_output: Matrix<Firing>) -> Matrix<Firing> {
    relu_output
        .0
        .mapv_inplace(|x| if x > 0.0 { 1.0 } else { 0.0 });
    relu_output
}

fn leaky_relu_derivative(mut leaky_relu_output: Matrix<Firing>, slope: f64) -> Matrix<Firing> {
    leaky_relu_output
        .0
        .mapv_inplace(|x| if x > 0.0 { 1.0 } else { slope });
    leaky_relu_output
}

fn sigmoid_derivative(sigmoid_output: &Matrix<Firing>) -> Matrix<Firing> {
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
        let result = relu.activate(input);
        assert_eq!(result, arr2(&[[1.0, 0.0], [2.0, 0.0]]));

        let act_grad = relu.calculate_derivative(&result);
        assert_eq!(act_grad, arr2(&[[1.0, 0.0], [1.0, 0.0]]));
    }
}
