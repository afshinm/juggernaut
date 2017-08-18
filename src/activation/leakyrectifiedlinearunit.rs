use activation::Activation;

#[derive(Copy, Clone)]
pub struct LeakyRectifiedLinearUnit {
    alpha_gradient: f64,
}

impl LeakyRectifiedLinearUnit {
    pub fn new(alpha: f64) -> LeakyRectifiedLinearUnit {
        return LeakyRectifiedLinearUnit {
            alpha_gradient: alpha,
        };
    }
}

impl Activation for LeakyRectifiedLinearUnit {
    /// Calculates the LeakyRectifiedLinearUnit of input `x`
    fn calc(&self, x: f64) -> f64 {
        if x <= 0f64 {
            self.alpha_gradient * x
        } else {
            x
        }
    }

    /// Calculates the Derivative LeakyRectifiedLinearUnit of input `x`
    fn derivative(&self, x: f64) -> f64 {
        if x <= 0f64 {
            self.alpha_gradient
        } else {
            x
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Activation;
    use super::LeakyRectifiedLinearUnit;

    #[test]
    fn leakyrectifiedlinearunit_test() {
        let activation = LeakyRectifiedLinearUnit::new(0.01f64);
        assert_approx_eq!(activation.calc(3.4f64), 3.4f64);
        assert_approx_eq!(activation.calc(-3.4f64), -0.034f64);
    }

    #[test]
    fn leakyrectifiedlinearunit_derivative_test() {
        let activation = LeakyRectifiedLinearUnit::new(0.01f64);
        assert_approx_eq!(activation.derivative(-3.4f64), 0.01f64);
        assert_approx_eq!(activation.derivative(3.4f64), 3.4f64);
    }
}
