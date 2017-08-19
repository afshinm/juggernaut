use activation::Activation;

#[derive(Copy, Clone)]
pub struct Identity;

impl Identity {
    pub fn new() -> Identity {
        return Identity;
    }
}

impl Activation for Identity {
    /// Calculates the Identity of input `x`
    fn calc(&self, x: f64) -> f64 {
        x
    }

    /// Calculates the Derivative Identity of input `x`
    fn derivative(&self, _: f64) -> f64 {
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
