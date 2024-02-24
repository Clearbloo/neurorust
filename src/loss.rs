use ndarray::{Array2, Zip};

/// Enum to represent different types of loss functions.
#[derive(Debug, Clone)]
pub enum Metric {
    MSE,
    MAE,
}

impl Metric {
    #[must_use] pub fn calculate_loss(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> f64 {
        match self {
            Self::MSE => mse(predictions, targets),
            Self::MAE => mae(predictions, targets),
        }
    }
    #[must_use] pub fn calculate_gradient(
        &self,
        predictions: &Array2<f64>,
        targets: &Array2<f64>,
    ) -> Array2<f64> {
        match self {
            Self::MSE => mse_gradient(predictions, targets),
            Self::MAE => mae_gradient(predictions, targets),
        }
    }
}

#[allow(clippy::cast_precision_loss)]
fn mse(predictions: &Array2<f64>, targets: &Array2<f64>) -> f64 {
    let n = predictions.shape()[1] as f64;
    Zip::from(predictions)
        .and(targets)
        .fold(0.0, |acc, &pred, &target| {
            (pred - target).mul_add(pred - target, acc)
        })
        / n
}
fn mse_gradient(predictions: &Array2<f64>, targets: &Array2<f64>) -> Array2<f64> {
    let n = predictions.shape()[1] as f64;
    2.0 * (predictions - targets) / n
}

#[allow(clippy::cast_precision_loss)]
fn mae(predictions: &Array2<f64>, targets: &Array2<f64>) -> f64 {
    let n = predictions.shape()[1] as f64;
    Zip::from(predictions)
        .and(targets)
        .fold(0.0, |acc, &pred, &target| acc + (pred - target).abs())
        / n
}
fn mae_gradient(predictions: &Array2<f64>, targets: &Array2<f64>) -> Array2<f64> {
    let n = predictions.shape()[1] as f64;
    (predictions - targets).map(|x| x.signum() / n)
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod test_loss_functions {
    use ndarray::arr2;

    use super::*;

    #[test]
    fn test_mse() {
        let predictions = Array2::from_elem([1, 4], 6.0);
        let targets = Array2::from_elem([1, 4], 2.0);

        let mse = Metric::MSE;
        let result = mse.calculate_loss(&predictions, &targets);
        println!("MSE Loss: {result}");
        // The differences are all 4.0, so avg MSE = 16.0
        assert_eq!(result, 16.0);

        let grad = mse.calculate_gradient(&predictions, &targets);
        assert_eq!(grad, Array2::from_elem([1, 4], 2.0));

        let predictions = arr2(&[[1., 2., 3.]]);
        let targets = arr2(&[[1., 1., 1.]]);
        assert!(mse.calculate_loss(&predictions, &targets) == (0. + 1. + 4.) / 3.);
    }

    #[test]
    fn test_mae() {
        let predictions = Array2::from_elem([1, 5], 2.0);
        let targets = Array2::from_elem([1, 5], 1.0);

        let mae = Metric::MAE;
        let result = mae.calculate_loss(&predictions, &targets);
        println!("MAE Loss: {result}");
        assert_eq!(result, 1.0);

        let grad = mae.calculate_gradient(&predictions, &targets);
        assert_eq!(grad, Array2::from_elem([1, 5], 0.2));
    }
}
