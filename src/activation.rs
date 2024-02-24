use crate::gradient::Grad;
use ndarray::Array2;

pub trait Activation: Grad {
    fn activate(&self, input: Array2<f64>) -> Array2<f64>;
}

pub struct ReLU;
pub struct Sigmoid;
pub struct LeakyReLU;

impl Grad for ReLU {
    fn grad(&self, output: &Array2<f64>) -> Array2<f64> {
        output.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 })
    }
}
impl Activation for ReLU {
    fn activate(&self, input: Array2<f64>) -> Array2<f64> {
        input.mapv(|x| if x > 0.0 { x } else { 0.0 })
    }
}

impl Grad for LeakyReLU {
    fn grad(&self, output: &Array2<f64>) -> Array2<f64> {
        output.mapv(|x| if x > 0.0 { 1.0 } else { 0.1 })
    }
}
impl Activation for LeakyReLU {
    fn activate(&self, input: Array2<f64>) -> Array2<f64> {
        input.mapv(|x| if x > 0.0 { x } else { 0.1 * x })
    }
}

impl Grad for Sigmoid {
    fn grad(&self, output: &Array2<f64>) -> Array2<f64> {
        output.mapv(|x| x * (1.0 - x))
    }
}
impl Activation for Sigmoid {
    fn activate(&self, input: Array2<f64>) -> Array2<f64> {
        input.mapv(|x| 1.0 / (1.0 + (-x).exp()))
    }
}
