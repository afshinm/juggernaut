use activation::Activation;

#[derive(Copy, Clone)]
pub struct Sigmoid;

impl Sigmoid {
    pub fn new() -> Sigmoid {
        return Sigmoid;
    }
}

impl Activation for Sigmoid {
    /// Calculates the Sigmoid of input `x`
    fn calc(&self, x: f64) -> f64 {
        1f64 / (1f64 + (-x).exp())
    }

    /// Calculates the Derivative Sigmoid of input `x`
    fn derivative(&self, x: f64) -> f64 {
        x * (1f64 - x)
    }
}

#[cfg(test)]
mod tests {
    use super::Activation;
    use super::Sigmoid;

    #[test]
    fn sigmoid_test() {
        let activation = Sigmoid::new();
        assert_approx_eq!(activation.calc(5f64), 0.9933071490f64);
    }

    #[test]
    fn sigmoid_derivative_test() {
        let activation = Sigmoid::new();
        assert_approx_eq!(activation.derivative(5f64), -20f64);
    }

}
