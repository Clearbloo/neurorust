use ndarray::Array2;

pub trait Activate {
    fn activate(&self, input: &Array2<f64>) -> Array2<f64>;
    fn calculate_gradient(&self, output_gradient: &Array2<f64>) -> Array2<f64>;
}

#[derive(Debug)]
pub enum Activation {
    ReLU,
    Sigmoid,
    LeakyReLU(f64),
}

impl Activate for Activation {
    // Implement the trait for the enum
    fn activate(&self, x: &Array2<f64>) -> Array2<f64> {
        match self {
            Self::ReLU => relu(x),
            Self::LeakyReLU(slope) => leaky_relu(x, *slope),
            Self::Sigmoid => sigmoid(x),
        }
    }
    fn calculate_gradient(&self, output_gradient: &Array2<f64>) -> Array2<f64> {
        match self {
            Self::ReLU => relu_gradient(output_gradient),
            Self::LeakyReLU(slope) => leaky_relu_gradient(output_gradient, *slope),
            Self::Sigmoid => sigmoid_gradient(output_gradient),
        }
    }
}

// Activation functions
fn relu(input: &Array2<f64>) -> Array2<f64> {
    input.map(|x| if x > &0.0 { *x } else { 0.0 })
}

fn leaky_relu(input: &Array2<f64>, slope: f64) -> Array2<f64> {
    input.map(|x| if x > &0.0 { *x } else { slope * x })
}

fn sigmoid(input: &Array2<f64>) -> Array2<f64> {
    input.map(|x| 1.0 / (1.0 + (-x).exp()))
}

// FIXME - I think these gradients are wrong. Need to think whether I want this to return just the activation gradient
// or the full gradient. As in just that part of the chain rule (du/dx) or the dy/du du/dx
fn relu_gradient(output_grad: &Array2<f64>) -> Array2<f64> {
    output_grad.mapv(|x| if x > 0.0 { x } else { 0.0 })
}

fn leaky_relu_gradient(output_grad: &Array2<f64>, slope: f64) -> Array2<f64> {
    output_grad.mapv(|x| if x > 0.0 { x } else { slope * x })
}

fn sigmoid_gradient(output_grad: &Array2<f64>) -> Array2<f64> {
    output_grad.mapv(|x| x * (1.0 - x) * x)
}

#[cfg(test)]
mod test_activations {
    use super::Activate;
    use super::Activation;
    use ndarray::arr2;

    #[test]
    fn test_relu() {
        let relu = Activation::ReLU;
        let input = arr2(&[[1.0, -2.0], [2.0, -3.0]]);
        let result = relu.activate(&input);
        assert_eq!(result, arr2(&[[1.0, 0.0], [2.0, 0.0]]));

        let act_grad = relu.calculate_gradient(&result);
        assert_eq!(act_grad, arr2(&[[1.0, 0.0], [2.0, 0.0]]));
    }
}
