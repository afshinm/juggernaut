extern crate rand;

use rand::{IsaacRng, Rng, SeedableRng};
use std::sync::Mutex;

#[derive(Debug, Clone, PartialEq)]
pub struct Matrix(Vec<Vec<f64>>);

pub trait MatrixTrait {
    fn zero(m: usize, n: usize) -> Self;
    fn random(m: usize, n: usize) -> Self;
    fn from_vec(v: &Vec<f64>) -> Self;
    fn generate(m: usize, n: usize, f: &Fn(usize, usize) -> f64) -> Self;
    fn generate_by_row(m: usize, n: usize, f: &Fn(usize) -> Vec<f64>) -> Self;
    fn row(&self, n: usize) -> &Vec<f64>;
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
    fn get(&self, m: usize, n: usize) -> f64;
    fn dot(&self, b: &Matrix) -> Matrix;
    fn transpose(&self) -> Matrix;
    fn map(&self, f: &Fn(f64, usize, usize) -> f64) -> Matrix;
    fn map_row(&self, f: &Fn(Vec<f64>) -> Vec<f64>) -> Matrix;
    fn body(&self) -> &Vec<Vec<f64>>;
}

impl MatrixTrait for Matrix {
    /// Returns a vector with `m` rows and `n` columns
    ///
    /// Example:
    ///
    /// ```
    /// # #[macro_use] extern crate juggernaut;
    /// # fn main() {
    /// use juggernaut::matrix::Matrix;
    /// use juggernaut::matrix::MatrixTrait;
    ///
    /// let matrix = Matrix::generate(2, 2, &|_,_| 2f64);
    /// # }
    /// ```
    fn generate(m: usize, n: usize, f: &Fn(usize, usize) -> f64) -> Matrix {
        let mut mtx: Vec<Vec<f64>> = Vec::with_capacity(m);

        for i in 0..m {
            let mut row: Vec<f64> = Vec::with_capacity(n);

            for j in 0..n {
                row.push(f(i, j));
            }

            mtx.push(row);
        }

        Matrix(mtx)
    }

    /// Returns a new Matrix with `m` rows and `n` columns
    ///
    /// Works exactly same as `generate` but accepts a Fn that returns each row of the matrix
    /// instead of each member of the matrix
    fn generate_by_row(m: usize, n: usize, f: &Fn(usize) -> Vec<f64>) -> Matrix {
        let mut mtx: Vec<Vec<f64>> = Vec::with_capacity(m);

        for i in 0..m {
            let row = f(i);

            assert!(row.len() == n);

            mtx.push(row);
        }

        Matrix(mtx)
    }

    /// Returns a vector with `m` rows and `n` columns with elements of 0
    fn zero(m: usize, n: usize) -> Matrix {
        Matrix::generate(m, n, &|_, _| 0f64)
    }

    /// Returns a vector with `m` rows and `n` columns with random elements
    fn random(m: usize, n: usize) -> Matrix {
        // TODO (afshinm): is this correct to set an array with one element as a seed?
        let numbers = Mutex::new((0..).scan(
            IsaacRng::from_seed(&[42]),
            |rng, _| Some(rng.next_f64()),
        ));

        Matrix::generate(m, n, &|_, _| numbers.lock().unwrap().next().unwrap())
    }

    /// Generates Matrix from a vector
    fn from_vec(v: &Vec<f64>) -> Matrix {
        Matrix::generate(1, v.len(), &|_, n| v[n])
    }

    /// Returns a row of Matrix
    fn row(&self, n: usize) -> &Vec<f64> {
        &self.0[n]
    }

    /// Number of the Matrix rows
    fn rows(&self) -> usize {
        self.0.len()
    }

    /// Number of the Matrix columns
    fn cols(&self) -> usize {
        self.0[0].len()
    }

    /// Returns the element in the position M,N
    fn get(&self, m: usize, n: usize) -> f64 {
        assert!(self.rows() > m && self.cols() > n);

        self.0[m][n]
    }

    /// Multiplication with Matrix
    fn dot(&self, b: &Matrix) -> Matrix {
        assert_eq!(self.cols(), b.rows());

        let mut result: Matrix = Matrix::zero(self.rows(), b.cols());

        for (m, row) in self.0.iter().enumerate() {
            for n in 0usize..b.cols() {
                let mut cell_result: f64 = 0f64;

                for (k, row_cell) in row.iter().enumerate() {
                    // row of the first Matrix X col of the second Matrix
                    cell_result += row_cell * b.get(k, n);
                }

                result.0[m][n] = cell_result;
            }
        }

        result
    }

    /// Transpose of a Matrix
    fn transpose(&self) -> Matrix {
        return Matrix::generate(self.cols(), self.rows(), &|m, n| self.get(n, m));
    }

    /// Map
    fn map(&self, f: &Fn(f64, usize, usize) -> f64) -> Matrix {
        return Matrix::generate(self.rows(), self.cols(), &|m, n| f(self.get(m, n), m, n));
    }

    /// Map for each row of Matrix
    fn map_row(&self, f: &Fn(Vec<f64>) -> Vec<f64>) -> Matrix {
        return Matrix::generate_by_row(self.rows(), self.cols(), &|m| f(self.row(m).to_vec()));
    }

    /// To get the first element of the Matrix (Vec<Vec<_>>)
    fn body(&self) -> &Vec<Vec<f64>> {
        return &self.0;
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

    #[test]
    fn random_matrix_get() {
        let test = Matrix::random(2, 2);

        assert_approx_eq!(test.get(0, 1), test.0[0][1]);
        assert_approx_eq!(test.get(1, 0), test.0[1][0]);
        assert_approx_eq!(test.get(1, 1), test.0[1][1]);
        assert_approx_eq!(test.get(0, 0), test.0[0][0]);
    }

    #[test]
    fn random_mul_test1() {
        let a = Matrix(vec![vec![1f64, 2f64], vec![3f64, 4f64]]);
        let b = Matrix(vec![vec![2f64, 0f64], vec![1f64, 2f64]]);
        let result = Matrix(vec![vec![4f64, 4f64], vec![10f64, 8f64]]);

        assert_eq!(a.dot(&b), result);
    }

    #[test]
    fn random_mul_test2() {
        let a = Matrix(vec![vec![1f64, 2f64], vec![3f64, 4f64]]);
        let b = Matrix(vec![vec![2f64, 0f64], vec![1f64, 2f64]]);
        let result = Matrix(vec![vec![2f64, 4f64], vec![7f64, 10f64]]);

        assert_eq!(b.dot(&a), result);
    }

    #[test]
    fn random_mul_test3() {
        let a = Matrix(vec![vec![1f64, 2f64, 3f64], vec![4f64, 5f64, 6f64]]);
        let b = Matrix(vec![
            vec![7f64, 8f64],
            vec![9f64, 10f64],
            vec![11f64, 12f64],
        ]);
        let result = Matrix(vec![vec![58f64, 64f64], vec![139f64, 154f64]]);

        assert_eq!(a.dot(&b), result);
    }

    #[test]
    fn random_mul_test4() {
        let a = Matrix(vec![vec![1f64, 0f64]]);

        let b = Matrix(vec![vec![3f64, 4f64, 5f64], vec![2f64, 3f64, 5f64]]);

        let result = Matrix(vec![vec![3f64, 4f64, 5f64]]);

        assert_eq!(a.dot(&b), result);
    }

    #[test]
    fn from_vec() {
        let v: Vec<f64> = vec![5f64, 1f64];

        let test = Matrix::from_vec(&v);

        let result = Matrix(vec![vec![5f64, 1f64]]);

        assert_eq!(test, result);
    }

    #[test]
    fn transpose() {
        let a = Matrix(vec![
            vec![4f64, 7f64, 2f64, 1f64],
            vec![3f64, 9f64, 8f64, 6f64],
        ]);
        let b = Matrix(vec![
            vec![4f64, 3f64],
            vec![7f64, 9f64],
            vec![2f64, 8f64],
            vec![1f64, 6f64],
        ]);

        assert_eq!(a.transpose(), b);
    }

    #[test]
    fn body() {
        let a = Matrix(vec![
            vec![4f64, 7f64, 2f64, 1f64],
            vec![3f64, 9f64, 8f64, 6f64],
        ]);

        let b = vec![vec![4f64, 7f64, 2f64, 1f64], vec![3f64, 9f64, 8f64, 6f64]];

        assert_eq!(a.body(), &b);
    }

    #[test]
    fn generate_by_row() {
        let a = Matrix(vec![
            vec![4f64, 7f64, 2f64, 1f64],
            vec![4f64, 7f64, 2f64, 1f64],
        ]);

        let b = Matrix::generate_by_row(2, 4, &|m| vec![4f64, 7f64, 2f64, 1f64]);

        assert_eq!(a, b);
    }

    #[test]
    fn generate_by_row_variable_rows() {
        let a = Matrix(vec![
            vec![0f64, 0f64, 0f64, 0f64],
            vec![1f64, 1f64, 1f64, 1f64],
            vec![2f64, 2f64, 2f64, 2f64],
        ]);

        let b = Matrix::generate_by_row(3, 4, &|m| vec![m as f64, m as f64, m as f64, m as f64]);

        assert_eq!(a, b);
    }

    #[test]
    fn map_row_variable_rows() {
        let a = Matrix(vec![
            vec![4f64, 7f64, 2f64, 1f64],
            vec![4f64, 7f64, 2f64, 1f64],
            vec![4f64, 7f64, 2f64, 1f64],
        ]);

        let b = a.map_row(&|row| vec![4f64, 7f64, 2f64, 1f64]);

        assert_eq!(a, b);
    }
}
