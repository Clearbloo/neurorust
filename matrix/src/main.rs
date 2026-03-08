use matrix::Matrix;

fn main() {
    let a = Matrix::new([[1, 4], [2, 3], [7, 8]]);
    let b = Matrix::new([[2, 3, 2], [7, 8, 2]]);
    let c = a * b;
    dbg!(c);
}
