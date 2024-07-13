// Loss functions
// TODO - Convert these to enums
use ndarray::{Array2, Zip};

pub trait Loss {
    fn calculate_loss(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> f64;
    fn calculate_gradient(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> Array2<f64>;
}

/// Enum to represent different types of loss functions.
pub struct MeanSquaredError;
pub struct MeanAbsoluteError;

impl Loss for MeanSquaredError {
    fn calculate_loss(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> f64 {
        let n = predictions.len() as f64;
        Zip::from(predictions)
            .and(targets)
            .fold(0.0, |acc, &pred, &target| acc + (pred - target).powi(2))
            / n
    }
    fn calculate_gradient(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> Array2<f64> {
        2.0 * (predictions - targets)
    }
}
impl Loss for MeanAbsoluteError {
    fn calculate_loss(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> f64 {
        let n = predictions.len() as f64;
        Zip::from(predictions)
            .and(targets)
            .fold(0.0, |acc, &pred, &target| acc + (pred - target).abs())
            / n
    }
    fn calculate_gradient(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> Array2<f64> {
        (predictions - targets).map(|x| x.signum())
    }
}

#[cfg(test)]
mod test_loss_functions {

    use super::*;

    #[test]
    fn test_mse() {
        let predictions = Array2::from_elem([1, 5], 6.0);
        let targets = Array2::from_elem([1, 5], 1.0);

        let mse = MeanSquaredError;
        let result = mse.calculate_loss(&predictions, &targets);
        println!("MSE Loss: {}", result);
        // The average difference is 5.0, so MSE = 25.0
        assert_eq!(result, 25.0);

        let grad = mse.calculate_gradient(&predictions, &targets);
        assert_eq!(grad, Array2::from_elem([1, 5], 10.0))
    }

    #[test]
    fn test_mae() {
        let predictions = Array2::from_elem([1, 5], 2.0);
        let targets = Array2::from_elem([1, 5], 1.0);

        let mae = MeanAbsoluteError;
        let result = mae.calculate_loss(&predictions, &targets);
        println!("MAE Loss: {}", result);
        assert_eq!(result, 1.0);

        let grad = mae.calculate_gradient(&predictions, &targets);
        assert_eq!(grad, Array2::from_elem([1, 5], 1.0))
    }
}
