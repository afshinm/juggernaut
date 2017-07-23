/// Activation functions
pub trait Activation {
    fn new() -> Self;
    // the function itself
    fn calc(&self, x: f64) -> f64;
    // Derivative
    fn derivative(&self, x: f64) -> f64;
}

pub struct Sigmoid;

impl Activation for Sigmoid {
    fn new() -> Sigmoid {
        return Sigmoid;
    }

    /// Calculates the Sigmoid of input `x`
    fn calc(&self, x: f64) -> f64 {
        1f64 / (1f64 + (-x).exp())
    }

    /// Calculates the Derivative Sigmoid of input `x`
    fn derivative(&self, x: f64) -> f64 {
        x * (1f64 - x)
    }
}

pub struct Identity;

impl Activation for Identity {
    fn new() -> Identity {
        return Identity;
    }

    /// Calculates the Identity of input `x`
    fn calc(&self, x: f64) -> f64 {
        x
    }

    /// Calculates the Derivative Identity of input `x`
    fn derivative(&self, x: f64) -> f64 {
        1f64
    }
}

#[cfg(test)]
mod tests {
    use super::Activation;
    use super::Sigmoid;
    use super::Identity;

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

    #[test]
    fn identity_test() {
        let activation = Identity::new();
        assert_approx_eq!(activation.calc(5f64), 5f64);
    }

    #[test]
    fn identity_derivative_test() {
        let activation = Identity::new();
        assert_approx_eq!(activation.derivative(15f64), 1f64);
    }
}

