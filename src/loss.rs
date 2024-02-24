// Loss functions
use ndarray::{Array2, Zip};

pub trait Loss {
    fn calculate(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> f64;
}

/// Enum to represent different types of loss functions.
pub enum LossFunction {
    MeanSquaredError,
    MeanAbsoluteError,
}

impl Loss for LossFunction {
    fn calculate(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> f64 {
        match self {
            LossFunction::MeanSquaredError => {
                let n = predictions.len() as f64;
                Zip::from(predictions)
                    .and(targets)
                    .fold(0.0, |acc, &pred, &target| acc + (pred - target).powi(2))
                    / n
            }
            LossFunction::MeanAbsoluteError => {
                let n = predictions.len() as f64;
                Zip::from(predictions)
                    .and(targets)
                    .fold(0.0, |acc, &pred, &target| acc + (pred - target).abs())
                    / n
            }
        }
    }
}

#[cfg(test)]
mod test_loss_functions {
    use super::*;

    #[test]
    fn test_mse() {
        let predictions = Array2::from_elem([1, 5], 3.0);
        let targets = Array2::from_elem([1, 5], 1.0);

        let mse = LossFunction::MeanSquaredError;
        let result = mse.calculate(&predictions, &targets);
        println!("MSE Loss: {}", result);
        assert_eq!(result, 4.0);
    }

    #[test]
    fn test_mae() {
        let predictions = Array2::from_elem([1, 5], 2.0);
        let targets = Array2::from_elem([1, 5], 1.0);

        let mae = LossFunction::MeanAbsoluteError;
        let result = mae.calculate(&predictions, &targets);
        println!("MAE Loss: {}", result);
        assert_eq!(result, 1.0)
    }
}
