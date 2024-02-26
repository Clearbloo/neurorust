use ndarray::Array2;

pub trait Optimization {
    fn apply_updates(&self, weight_updates: Vec<Array2<f64>>, bias_updates: Vec<Array2<f64>>);
}

pub struct Adam;

impl Optimization for Adam {
    fn apply_updates(&self, weight_updates: Vec<Array2<f64>>, bias_updates: Vec<Array2<f64>>) {
        println!("Weight updates: {:?}", weight_updates);
        println!("Bias updates: {:?}", bias_updates);
        todo!()
    }
}
