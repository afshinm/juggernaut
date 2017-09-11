use std::f64;
use activation::Activation;

#[derive(Copy, Clone)]
pub struct HyperbolicTangent;

impl HyperbolicTangent {
    pub fn new() -> HyperbolicTangent {
        return HyperbolicTangent;
    }
}

impl Activation for HyperbolicTangent {
    /// Calculates the tanh of input `x`
    fn calc(&self, x: Vec<f64>) -> Vec<f64> {
        x.iter().map(|n| n.tanh()).collect::<Vec<_>>()
    }

    /// Calculates the Derivative tanh of input `x`
    fn derivative(&self, x: Vec<f64>) -> Vec<f64> {
        x.iter()
            .map(|n| {
                let tanh_factor = n.tanh();
                1f64 - (tanh_factor * tanh_factor)
            })
            .collect::<Vec<_>>()
    }
}


#[cfg(test)]
mod tests {
    use super::Activation;
    use super::HyperbolicTangent;

    #[test]
    fn tanh_test() {
        let activation = HyperbolicTangent::new();
        assert_approx_eq!(activation.calc(vec![3f64])[0], 0.995054754f64);
    }

    #[test]
    fn tanh_derivative_test() {
        let activation = HyperbolicTangent::new();
        assert_approx_eq!(activation.derivative(vec![3f64])[0], 0.0098660372f64);
    }
}
