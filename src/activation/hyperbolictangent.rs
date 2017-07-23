use std::f64;
use activation::Activation;

pub struct HyperbolicTangent;

impl Activation for HyperbolicTangent {
    fn new() -> HyperbolicTangent {
        return HyperbolicTangent;
    }

    /// Calculates the tanh of input `x`
    fn calc(&self, x: f64) -> f64 {
        x.tanh()
    }

    /// Calculates the Derivative tanh of input `x`
    fn derivative(&self, x: f64) -> f64 {
        let tanh_factor = x.tanh();
        1f64 - (tanh_factor * tanh_factor)
    }
}


#[cfg(test)]
mod tests {
    use super::Activation;
    use super::HyperbolicTangent;

    #[test]
    fn tanh_test() {
        let activation = HyperbolicTangent::new();
        assert_approx_eq!(activation.calc(3f64), 0.995054754f64);
    }

    #[test]
    fn tanh_derivative_test() {
        let activation = HyperbolicTangent::new();
        assert_approx_eq!(activation.derivative(3f64), 0.0098660372f64);
    }
}
