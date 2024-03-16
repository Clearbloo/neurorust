// use crate::gradient::Grad;
use ndarray::Array2;

pub trait Activation {
    fn activate(&self, input: Array2<f64>) -> Array2<f64>;
    fn calculate_gradient(&self, output: &Array2<f64>) -> Array2<f64>;
}

pub struct ReLU;
pub struct Sigmoid;
pub struct LeakyReLU;

impl Activation for ReLU {
    fn activate(&self, input: Array2<f64>) -> Array2<f64> {
        input.mapv(|x| if x > 0.0 { x } else { 0.0 })
    }
    fn calculate_gradient(&self, output: &Array2<f64>) -> Array2<f64> {
        output.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 })
    }
}

impl Activation for LeakyReLU {
    fn activate(&self, input: Array2<f64>) -> Array2<f64> {
        input.mapv(|x| if x > 0.0 { x } else { 0.1 * x })
    }
    fn calculate_gradient(&self, output: &Array2<f64>) -> Array2<f64> {
        output.mapv(|x| if x > 0.0 { 1.0 } else { 0.1 })
    }
}

impl Activation for Sigmoid {
    fn activate(&self, input: Array2<f64>) -> Array2<f64> {
        input.mapv(|x| 1.0 / (1.0 + (-x).exp()))
    }
    fn calculate_gradient(&self, output: &Array2<f64>) -> Array2<f64> {
        output.mapv(|x| x * (1.0 - x))
    }
}

#[cfg(test)]
mod test_activations {
    use super::{Activation, ReLU};
    use ndarray::arr2;

    #[test]
    fn test_relu() {
        let relu = ReLU;
        let input = arr2(&[[1.0, -2.0], [2.0, -3.0]]);
        let result = relu.activate(input);
        assert_eq!(result, arr2(&[[1.0, 0.0], [2.0, 0.0]]));

        let act_grad = relu.calculate_gradient(&result);
        assert_eq!(act_grad, arr2(&[[1.0, 0.0], [1.0, 0.0]]));
    }
}
