use ndarray::{Array2, Zip};

pub fn arrays_are_close(a: &Array2<f64>, b: &Array2<f64>, tolerance: f64) -> bool {
    if a.shape() != b.shape() {
        return false;
    }

    Zip::from(a)
        .and(b)
        .fold(true, |acc, &a, &b| acc && (a - b).abs() <= tolerance)
}
// Min-Max Scaling
pub fn min_max_scale(input: &Array2<f64>) -> Array2<f64> {
    let min = input.fold(f64::INFINITY, |a, &b| a.min(b));
    let max = input.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    (input - min) / (max - min)
}
