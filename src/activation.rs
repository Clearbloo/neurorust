use core::fmt::Debug;
use ndarray::Array2;

#[derive(Debug, Clone)]
pub enum Activation {
    Linear,
    ReLU,
    Sigmoid,
    LeakyReLU(f64),
}

impl Activation {
    pub fn activate(&self, x: &Array2<f64>) -> Array2<f64> {
        match self {
            Self::Linear => x.clone(),
            Self::ReLU => relu(x),
            Self::LeakyReLU(slope) => leaky_relu(x, *slope),
            Self::Sigmoid => sigmoid(x),
        }
    }
    /// Returns the gradient of the activation function. Same shape as the
    /// output_gradient parameter
    pub fn calculate_derivative(&self, pre_activation_output: &Array2<f64>) -> Array2<f64> {
        match self {
            Self::Linear => {
                let x_dim = pre_activation_output.shape()[0];
                let y_dim = pre_activation_output.shape()[1];
                Array2::from_elem((x_dim, y_dim), 1.)
            }
            Self::ReLU => relu_derivative(pre_activation_output),
            Self::LeakyReLU(slope) => leaky_relu_derivative(pre_activation_output, *slope),
            Self::Sigmoid => sigmoid_derivative(pre_activation_output),
        }
    }
}

// Activation functions
fn relu(input: &Array2<f64>) -> Array2<f64> {
    input.mapv(|x| if x > 0.0 { x } else { 0.0 })
}

fn leaky_relu(input: &Array2<f64>, slope: f64) -> Array2<f64> {
    input.mapv(|x| if x > 0.0 { x } else { slope * x })
}

fn sigmoid(input: &Array2<f64>) -> Array2<f64> {
    input.mapv(|x| 1.0 / (1.0 + (-x).exp()))
}

fn relu_derivative(relu_output: &Array2<f64>) -> Array2<f64> {
    relu_output.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 })
}

fn leaky_relu_derivative(leaky_relu_output: &Array2<f64>, slope: f64) -> Array2<f64> {
    leaky_relu_output.mapv(|x| if x > 0.0 { 1.0 } else { slope })
}

fn sigmoid_derivative(sigmoid_output: &Array2<f64>) -> Array2<f64> {
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
