use std::f64;

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

pub struct SoftPlus;

impl Activation for SoftPlus {
    fn new() -> SoftPlus {
        return SoftPlus;
    }

    /// Calculates the SoftPlus of input `x`
    fn calc(&self, x: f64) -> f64 {
        (1f64 + x.exp()).ln()
    }

    /// Calculates the Derivative SoftPlus of input `x`
    fn derivative(&self, x: f64) -> f64 {
        1f64 / (1f64 + (-x).exp())
    }
}

pub struct RectifiedLinearUnit;

impl Activation for RectifiedLinearUnit {
    fn new() -> RectifiedLinearUnit {
        return RectifiedLinearUnit;
    }

    /// Calculates the RectifiedLinearUnit of input `x`
    fn calc(&self, x: f64) -> f64 {
        if x <= 0f64 {
            0f64
        } else {
            x
        }
    }

    /// Calculates the Derivative RectifiedLinearUnit of input `x`
    fn derivative(&self, x: f64) -> f64 {
        if x <= 0f64 {
            0f64
        } else {
            x
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Activation;
    use super::Sigmoid;
    use super::Identity;
    use super::HyperbolicTangent;
    use super::SoftPlus;
    use super::RectifiedLinearUnit;

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

    #[test]
    fn softplus_test() {
        let activation = SoftPlus::new();
        assert_approx_eq!(activation.calc(-1f64), 0.3132616875f64);
    }

    #[test]
    fn softplus_derivative_test() {
        let activation = SoftPlus::new();
        assert_approx_eq!(activation.derivative(-1f64), 0.2689414214f64);
    }

    #[test]
    fn rectifiedlinearunit_test() {
        let activation = RectifiedLinearUnit::new();
        assert_approx_eq!(activation.calc(3.4f64), 3.4f64);
        assert_approx_eq!(activation.calc(-3.4f64), 0f64);
    }

    #[test]
    fn rectifiedlinearunit_derivative_test() {
        let activation = RectifiedLinearUnit::new();
        assert_approx_eq!(activation.derivative(-3.4f64), 0f64);
        assert_approx_eq!(activation.derivative(3.4f64), 3.4f64);
    }
}

