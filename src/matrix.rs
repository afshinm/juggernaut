#[derive(Debug, Clone, PartialEq)]
pub struct Matrix(Vec<Vec<f64>>);

pub trait MatrixTrait {
    fn new(m: usize, n: usize) -> Self;
}

impl MatrixTrait for Matrix {
    /// Returns a vector with `m` rows and `n` columns
    fn new(m: usize, n: usize) -> Matrix {
        let mut mtx: Vec<Vec<f64>> = Vec::with_capacity(n);

        for _ in 0..n {
            let mut row: Vec<f64> = Vec::with_capacity(m);

            for _ in 0..m {
                row.push(0f64);
            }

            mtx.push(row);
        }

        Matrix(mtx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_matrix_test() {
        let test = Matrix(vec![vec![0f64, 0f64], vec![0f64, 0f64]]);
        assert_eq!(Matrix::new(2, 2), test);
    }
}
