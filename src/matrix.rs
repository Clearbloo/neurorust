use std::fmt;
use std::ops::Mul;
use std::sync::{Arc, Mutex};
use std::thread;

#[derive(Debug)]
pub enum MatrixError {
    DimensionMismatch,
    InvalidDimension,
}

impl fmt::Display for MatrixError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            MatrixError::DimensionMismatch => {
                write!(f, "Matrix dimensions are incompatible for multiplication")
            }
            MatrixError::InvalidDimension => write!(f, "Invalid matrix dimensions"),
        }
    }
}

impl std::error::Error for MatrixError {}

#[derive(Debug, Clone)]
pub struct Matrix {
    data: Vec<Vec<f64>>,
    rows: usize,
    cols: usize,
}

impl Matrix {
    /// Creates a new matrix with the given data
    ///
    /// # Arguments
    /// * `data` - A 2D vector containing the matrix elements
    ///
    /// # Returns
    /// * `Result<Matrix, MatrixError>` - A new Matrix if dimensions are valid
    pub fn new(data: Vec<Vec<f64>>) -> Result<Matrix, MatrixError> {
        if data.is_empty() || data.iter().any(|r| r.is_empty()) {
            return Err(MatrixError::InvalidDimension);
        }

        let rows = data.len();
        let cols = data[0].len();

        // Verify all rows have same length
        if data.iter().any(|row| row.len() != cols) {
            return Err(MatrixError::InvalidDimension);
        }

        Ok(Matrix { data, rows, cols })
    }

    /// Returns the dimensions of the matrix
    pub fn dimensions(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Multiplies two matrices in parallel
    ///
    /// # Arguments
    /// * `other` - The matrix to multiply with
    /// * `num_threads` - Number of threads to use for parallel computation
    ///
    /// # Returns
    /// * `Result<Matrix, MatrixError>` - The resulting matrix or an error
    pub fn multiply_parallel(
        &self,
        other: &Matrix,
        num_threads: usize,
    ) -> Result<Matrix, MatrixError> {
        // Check if matrices can be multiplied
        if self.cols != other.rows {
            return Err(MatrixError::DimensionMismatch);
        }

        let result_rows = self.rows;
        let result_cols = other.cols;

        // Create shared result matrix
        let result = Arc::new(Mutex::new(vec![vec![0.0; result_cols]; result_rows]));
        let mut handles = vec![];

        // Calculate rows per thread
        let rows_per_thread = result_rows.div_ceil(num_threads);

        // Spawn threads
        for thread_id in 0..num_threads {
            let start_row = thread_id * rows_per_thread;
            let end_row = (start_row + rows_per_thread).min(result_rows);

            // Skip if no rows to process
            if start_row >= result_rows {
                continue;
            }

            let result = Arc::clone(&result);
            let self_data = self.data.clone();
            let other_data = other.data.clone();

            handles.push(thread::spawn(move || {
                let mut local_result = vec![vec![0.0; result_cols]; end_row - start_row];

                // Compute assigned rows
                for i in 0..(end_row - start_row) {
                    for j in 0..result_cols {
                        let mut sum = 0.0;
                        for (k, other_row) in other_data.iter().enumerate() {
                            sum += self_data[start_row + i][k] * other_row[j];
                        }
                        local_result[i][j] = sum;
                    }
                }

                // Update shared result
                let mut result = result.lock().unwrap();
                for i in 0..(end_row - start_row) {
                    for j in 0..result_cols {
                        result[start_row + i][j] = local_result[i][j];
                    }
                }
            }));
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }

        // Create final matrix from result
        let result = Arc::try_unwrap(result).unwrap().into_inner().unwrap();

        Matrix::new(result)
    }
}

impl Mul for Matrix {
    type Output = Result<Matrix, color_eyre::eyre::Report>;
    fn mul(self, rhs: Self) -> Self::Output {
        let num_threads = std::thread::available_parallelism()?
            .get()
            .max(2)
            .div_ceil(2);
        Ok(self.multiply_parallel(&rhs, num_threads)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_creation() {
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let matrix = Matrix::new(data).unwrap();
        assert_eq!(matrix.dimensions(), (2, 2));
    }

    #[test]
    fn test_invalid_matrix() {
        let data = vec![vec![1.0, 2.0], vec![3.0]];
        assert!(Matrix::new(data).is_err());
    }

    #[test]
    fn test_parallel_multiplication() {
        let a = Matrix::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();

        let b = Matrix::new(vec![vec![5.0, 6.0], vec![7.0, 8.0]]).unwrap();

        let result = a.multiply_parallel(&b, 2).unwrap();
        let expected = vec![vec![19.0, 22.0], vec![43.0, 50.0]];

        assert_eq!(result.data, expected);
    }

    #[test]
    fn test_dimension_mismatch() {
        let a = Matrix::new(vec![vec![1.0, 2.0]]).unwrap();
        let b = Matrix::new(vec![vec![3.0], vec![4.0], vec![5.0]]).unwrap();
        assert!(matches!(
            a.multiply_parallel(&b, 2),
            Err(MatrixError::DimensionMismatch)
        ));
    }
}
