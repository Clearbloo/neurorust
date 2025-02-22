use color_eyre::eyre::{eyre, Report};
use ndarray::{Array2, Zip};

pub type Matrix = Array2<f64>;

#[must_use]
pub fn arrays_are_close(a: &Matrix, b: &Matrix, tolerance: f64) -> bool {
    if a.shape() != b.shape() {
        return false;
    }

    Zip::from(a)
        .and(b)
        .fold(true, |acc, &a, &b| acc && (a - b).abs() <= tolerance)
}
// Min-Max Scaling
#[must_use]
pub fn min_max_scale(input: &Matrix) -> Matrix {
    let min = input.fold(f64::INFINITY, |a, &b| a.min(b));
    let max = input.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    (input - min) / (max - min)
}

pub fn load_csv(fp: String) -> Result<csv::Reader<std::fs::File>, Report> {
    csv::Reader::from_path(fp).map_err(|e| eyre!(e))
}
