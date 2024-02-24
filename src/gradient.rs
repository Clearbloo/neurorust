use ndarray::Array2;
pub trait Grad {
    fn grad(&self, output: &Array2<f64>) -> Array2<f64>;
}
