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
    fn calc(&self, x: Vec<f64>) -> Vec<f64> {
        x
    }

    /// Calculates the Derivative Identity of input `x`
    fn derivative(&self, v: Vec<f64>) -> Vec<f64> {
        v.iter().map(|_| 1f64).collect::<Vec<_>>()
    }
}

#[cfg(test)]
mod tests {
    use super::Activation;
    use super::Identity;

    #[test]
    fn identity_test() {
        let activation = Identity::new();
        assert_approx_eq!(activation.calc(vec![5f64])[0], 5f64);
    }

    #[test]
    fn identity_derivative_test() {
        let activation = Identity::new();
        assert_approx_eq!(activation.derivative(vec![15f64])[0], 1f64);
    }
}
