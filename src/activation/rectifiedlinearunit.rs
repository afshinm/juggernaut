use activation::Activation;

#[derive(Copy, Clone)]
pub struct RectifiedLinearUnit;

impl RectifiedLinearUnit {
    pub fn new() -> RectifiedLinearUnit {
        return RectifiedLinearUnit;
    }
}

impl Activation for RectifiedLinearUnit {

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
