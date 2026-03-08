use crate::matrix::Matrix;
use color_eyre::eyre::{eyre, Report};

#[must_use]
pub fn arrays_are_close<T>(a: &Matrix<T>, b: &Matrix<T>, tolerance: f64) -> bool {
    if a.as_ref().shape() != b.as_ref().shape() {
        return false;
    }

    Zip::from(a.as_ref())
        .and(b.as_ref())
        .fold(true, |acc, &a, &b| acc && (a - b).abs() <= tolerance)
}
// Min-Max Scaling
#[must_use]
pub fn min_max_scale<T>(input: &Matrix<T>) -> Matrix<T> {
    let min = input.as_ref().fold(f64::INFINITY, |a, &b| a.min(b));
    let max = input.as_ref().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    (input - min) / (max - min)
}

pub fn load_csv(fp: String) -> Result<csv::Reader<std::fs::File>, Report> {
    csv::Reader::from_path(fp).map_err(|e| eyre!(e))
}
