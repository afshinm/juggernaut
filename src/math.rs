use std::ops::Mul;


/// Vector multiplication
trait Multiplication<T> {
    fn dot(&self, x: T) -> Self;
}

impl<T> Multiplication<T> for Vec<T> where T: Mul<Output = T> + Copy {
    fn dot(&self, x: T) -> Vec<T> {
        self.iter().map(|y| *y * x).collect()
    }
}

/// Activation functions
struct Activation {}

impl Activation {
    /// Calculates the Sigmoid of input `x`
    pub fn sigmoid(x: f64) -> f64 {
        1f64 / (1f64 + (-x).exp())
    }

    /// Calculates the Derivative Sigmoid of input `x`
    pub fn sigmoid_derivative(x: f64) -> f64 {
        x * (1f64 - x)
    }
}

#[cfg(test)]
mod tests {
    use super::Activation;

    #[test]
    fn sigmoid_test() {
        assert_approx_eq!(Activation::sigmoid(5f64), 0.9933071490f64);
    }

    #[test]
    fn sigmoid_derivative_test() {
        assert_approx_eq!(Activation::sigmoid_derivative(5f64), -20f64);
    }
}
