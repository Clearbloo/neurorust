use rayon::prelude::*;
use std::{
    iter::Sum,
    ops::{Add, Mul},
};

pub fn dot<'elem, T, const N: usize>(a: &'elem [T; N], b: &'elem [T; N]) -> T
where
    T: Add<Output = T> + Mul<Output = T> + Default + Sync + Send + Sum + Copy,
{
    a.par_iter()
        .zip(b)
        .fold(T::default, |acc, (x, y)| acc + (*x * *y))
        .sum()
}

#[derive(Debug, Clone)]
pub struct Matrix<T, const ROWS: usize, const COLS: usize> {
    data: [[T; COLS]; ROWS],
}

impl<T: Copy, const ROWS: usize, const COLS: usize> Matrix<T, ROWS, COLS> {
    pub fn new(data: [[T; COLS]; ROWS]) -> Self {
        Self { data }
    }
    // This might be slow
    pub fn col(&self, idx: usize) -> [T; ROWS] {
        std::array::from_fn(|i| self.data[i][idx])
    }
}

/// Matrix multiplication - note how dimensions must match at compile time!
/// (M × N) * (N × P) = (M × P)
impl<T, const M: usize, const N: usize, const P: usize> Mul<Matrix<T, N, P>> for Matrix<T, M, N>
where
    T: Copy + Default + Mul<Output = T> + Add<Output = T> + Sync + Send + Sum,
{
    type Output = Matrix<T, M, P>;

    fn mul(self, rhs: Matrix<T, N, P>) -> Self::Output {
        let mut result = [[T::default(); P]; M];
        result
            .par_iter_mut()
            .enumerate()
            .for_each(|(row_idx, row)| {
                let left_row = &self.data[row_idx];

                for col_idx in 0..N {
                    row[col_idx] = rhs
                        .col(col_idx)
                        .iter()
                        .zip(left_row)
                        .fold(T::default(), |acc, (x, y)| acc + *x * *y);
                }
                // row.par_iter_mut()
                //     .enumerate()
                //     .for_each(|(col_idx, elem)| *elem = dot(left_row, &rhs.col(col_idx)))
            });

        Matrix::new(result)
    }
}

// Addition - matrices must have same dimensions
impl<T, const ROWS: usize, const COLS: usize> Add for Matrix<T, ROWS, COLS>
where
    T: Copy + Add<Output = T>,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut result = self.data;
        for i in 0..ROWS {
            for j in 0..COLS {
                result[i][j] = result[i][j] + rhs.data[i][j];
            }
        }
        Matrix::new(result)
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn feature() {}
}
