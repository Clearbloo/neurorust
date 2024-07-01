use ndarray::Array2;

pub trait Activate {
    fn activate(&self, input: Array2<f64>) -> Array2<f64>;
    fn calculate_gradient(&self, output: &Array2<f64>) -> Array2<f64>;
}

#[derive(Debug)]
pub enum Activation {
    ReLU,
    Sigmoid,
    LeakyReLU(f64),
}

impl Activate for Activation {
    // Implement the trait for the enum
    fn activate(&self, x: Array2<f64>) -> Array2<f64> {
        match self {
            Activation::ReLU => relu(x),
            Activation::LeakyReLU(slope) => leaky_relu(x, *slope),
            Activation::Sigmoid => sigmoid(x),
        }
    }
    fn calculate_gradient(&self, output: &Array2<f64>) -> Array2<f64> {
        match self {
            Activation::ReLU => relu_gradient(output),
            Activation::LeakyReLU(slope) => leaky_relu_gradient(output, *slope),
            Activation::Sigmoid => sigmoid_gradient(output),
        }
    }
}

// Activation functions
fn relu(input: Array2<f64>) -> Array2<f64> {
    input.map(|x| if x > &0.0 { *x } else { 0.0 })
}

fn leaky_relu(input: Array2<f64>, slope: f64) -> Array2<f64> {
    input.map(|x| if x > &0.0 { *x } else { slope * x })
}

fn sigmoid(input: Array2<f64>) -> Array2<f64> {
    input.map(|x| 1.0 / (1.0 + (-x).exp()))
}

fn relu_gradient(input: &Array2<f64>) -> Array2<f64> {
    input.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 })
}

fn leaky_relu_gradient(output: &Array2<f64>, slope: f64) -> Array2<f64> {
    output.mapv(|x| if x > 0.0 { 1.0 } else { slope })
}

fn sigmoid_gradient(output: &Array2<f64>) -> Array2<f64> {
    output.mapv(|x| x * (1.0 - x))
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
        let result = relu.activate(input);
        assert_eq!(result, arr2(&[[1.0, 0.0], [2.0, 0.0]]));

        let act_grad = relu.calculate_gradient(&result);
        assert_eq!(act_grad, arr2(&[[1.0, 0.0], [1.0, 0.0]]));
    }
}
