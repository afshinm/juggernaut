use activation::Activation;

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
    use super::Identity;

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

