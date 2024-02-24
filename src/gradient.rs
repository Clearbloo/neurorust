use ndarray::Array2;
pub trait Grad {
    fn calculate_gradient(&self, input: &Array2<f64>, output: &Array2<f64>) -> Array2<f64>;
}
