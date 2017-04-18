extern crate rand;

use rand::Rng;

#[derive(Debug, Clone, PartialEq)]
pub struct Matrix(Vec<Vec<f64>>);

pub trait MatrixTrait {
    fn zero(m: usize, n: usize) -> Self;
    fn random(m: usize, n: usize) -> Self;
    fn generate(m: usize, n: usize, f: &Fn() -> f64) -> Self;
}

impl MatrixTrait for Matrix {
    /// Returns a vector with `m` rows and `n` columns
    fn generate(m: usize, n: usize, f: &Fn() -> f64) -> Matrix {
        let mut mtx: Vec<Vec<f64>> = Vec::with_capacity(n);

        for _ in 0..n {
            let mut row: Vec<f64> = Vec::with_capacity(m);

            for _ in 0..m {
                row.push(f());
            }

            mtx.push(row);
        }

        Matrix(mtx)
    }

    /// Returns a vector with `m` rows and `n` columns with elements of 0
    fn zero(m: usize, n: usize) -> Matrix {
        Matrix::generate(m, n, &|| 0f64)
    }

    /// Returns a vector with `m` rows and `n` columns with random elements
    fn random(m: usize, n: usize) -> Matrix {
        Matrix::generate(m, n, &|| rand::thread_rng().gen_range(0f64, 1f64))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_matrix_test() {
        let test = Matrix(vec![vec![0f64, 0f64], vec![0f64, 0f64]]);
        assert_eq!(Matrix::zero(2, 2), test);
    }

    #[test]
    fn random_matrix_test() {
        let test = Matrix::random(2, 2);

        assert_ne!(test.0[0][0], test.0[0][1]);
        assert_ne!(test.0[1][0], test.0[1][1]);
    }
}
