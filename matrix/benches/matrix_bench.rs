use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use matrix::Matrix;
use nalgebra as na;

fn bench_matrix_multiply(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_multiply");

    // Small matrices
    group.bench_function("custom_10x10", |b| {
        let m1: Matrix<f64, 10, 10> = Matrix::new([[1.0; 10]; 10]);
        let m2: Matrix<f64, 10, 10> = Matrix::new([[2.0; 10]; 10]);
        b.iter(|| black_box(m1.clone() * m2.clone()));
    });

    group.bench_function("nalgebra_10x10", |b| {
        let m1 = na::SMatrix::<f64, 10, 10>::from_element(1.0);
        let m2 = na::SMatrix::<f64, 10, 10>::from_element(2.0);
        b.iter(|| black_box(m1.clone() * m2.clone()));
    });

    // Medium matrices
    group.bench_function("custom_100x100", |b| {
        let m1: Matrix<f64, 100, 100> = Matrix::new([[1.0; 100]; 100]);
        let m2: Matrix<f64, 100, 100> = Matrix::new([[2.0; 100]; 100]);
        b.iter(|| black_box(m1.clone() * m2.clone()));
    });

    group.bench_function("nalgebra_100x100", |b| {
        let m1 = na::SMatrix::<f64, 100, 100>::from_element(1.0);
        let m2 = na::SMatrix::<f64, 100, 100>::from_element(2.0);
        b.iter(|| black_box(m1.clone() * m2.clone()));
    });

    group.finish();
}

fn bench_column_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("column_access");

    group.bench_function("custom_copy", |b| {
        let m: Matrix<f64, 100, 100> = Matrix::new([[1.0; 100]; 100]);
        b.iter(|| {
            let col = black_box(m.col(50));
            black_box(col)
        });
    });

    group.bench_function("custom_iter", |b| {
        let m: Matrix<f64, 100, 100> = Matrix::new([[1.0; 100]; 100]);
        b.iter(|| {
            let sum: f64 = black_box(m.col(50).iter().copied().sum());
            black_box(sum)
        });
    });

    group.bench_function("nalgebra", |b| {
        let m = na::SMatrix::<f64, 100, 100>::from_element(1.0);
        b.iter(|| {
            let col = black_box(m.column(50));
            black_box(col)
        });
    });

    group.finish();
}

criterion_group!(benches, bench_matrix_multiply, bench_column_access);
criterion_main!(benches);
