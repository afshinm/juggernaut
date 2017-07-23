pub mod sigmoid;
pub mod identity;
pub mod hyperbolictangent;
pub mod softplus;

pub use self::sigmoid::Sigmoid;
pub use self::identity::Identity;
pub use self::hyperbolictangent::HyperbolicTangent;
pub use self::softplus::SoftPlus;

/// Activation functions
pub trait Activation {
    fn new() -> Self;
    // the function itself
    fn calc(&self, x: f64) -> f64;
    // Derivative
    fn derivative(&self, x: f64) -> f64;
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
    use super::RectifiedLinearUnit;

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

