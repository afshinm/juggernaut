use std::f64;
use activation::Activation;

#[derive(Copy, Clone)]
pub struct SoftPlus;

impl SoftPlus {
    pub fn new() -> SoftPlus {
        return SoftPlus;
    }
}

impl Activation for SoftPlus {
    /// Calculates the SoftPlus of input `x`
    fn calc(&self, x: Vec<f64>) -> Vec<f64> {
        x.iter().map(|n| (1f64 + n.exp()).ln()).collect::<Vec<_>>()
    }

    /// Calculates the Derivative SoftPlus of input `x`
    fn derivative(&self, x: Vec<f64>) -> Vec<f64> {
        x.iter()
            .map(|n| 1f64 / (1f64 + (-n).exp()))
            .collect::<Vec<_>>()
    }
}

#[cfg(test)]
mod tests {
    use super::Activation;
    use super::SoftPlus;

    #[test]
    fn softplus_test() {
        let activation = SoftPlus::new();
        assert_approx_eq!(activation.calc(vec![-1f64])[0], 0.3132616875f64);
    }

    #[test]
    fn softplus_derivative_test() {
        let activation = SoftPlus::new();
        assert_approx_eq!(activation.derivative(vec![-1f64])[0], 0.2689414214f64);
    }
}
