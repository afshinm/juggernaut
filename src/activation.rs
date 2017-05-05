/// Activation functions
pub trait Activation {
    fn new() -> Self;
    // the function itself
    fn calc(x: f64) -> f64;
    // Derivative
    fn derivative(x: f64) -> f64;
}

pub struct Sigmoid {}

impl Activation for Sigmoid {
    fn new() -> Sigmoid {
        return Sigmoid{};
    }

    /// Calculates the Sigmoid of input `x`
    fn calc(x: f64) -> f64 {
        1f64 / (1f64 + (-x).exp())
    }

    /// Calculates the Derivative Sigmoid of input `x`
    fn derivative(x: f64) -> f64 {
        x * (1f64 - x)
    }
}

#[cfg(test)]
mod tests {
    use super::Activation;
    use super::Sigmoid;

    #[test]
    fn sigmoid_test() {
        assert_approx_eq!(Sigmoid::calc(5f64), 0.9933071490f64);
    }

    #[test]
    fn sigmoid_derivative_test() {
        assert_approx_eq!(Sigmoid::derivative(5f64), -20f64);
    }
}
