use activation::Activation;

#[derive(Copy, Clone)]
pub struct LeakyRectifiedLinearUnit {
    alpha_gradient: f64,
}

impl LeakyRectifiedLinearUnit {
    pub fn new(alpha: f64) -> LeakyRectifiedLinearUnit {
        return LeakyRectifiedLinearUnit { alpha_gradient: alpha };
    }
}

impl Activation for LeakyRectifiedLinearUnit {
    /// Calculates the LeakyRectifiedLinearUnit of input `x`
    fn calc(&self, x: Vec<f64>) -> Vec<f64> {
        x.iter()
            .map(|&n| if n <= 0f64 {
                self.alpha_gradient * n
            } else {
                n
            })
            .collect::<Vec<_>>()
    }

    /// Calculates the Derivative LeakyRectifiedLinearUnit of input `x`
    fn derivative(&self, x: Vec<f64>) -> Vec<f64> {
        x.iter()
            .map(|&n| if n <= 0f64 { self.alpha_gradient } else { n })
            .collect::<Vec<_>>()
    }
}

#[cfg(test)]
mod tests {
    use super::Activation;
    use super::LeakyRectifiedLinearUnit;

    #[test]
    fn leakyrectifiedlinearunit_test() {
        let activation = LeakyRectifiedLinearUnit::new(0.01f64);
        assert_approx_eq!(activation.calc(vec![3.4f64])[0], 3.4f64);
        assert_approx_eq!(activation.calc(vec![-3.4f64])[0], -0.034f64);
    }

    #[test]
    fn leakyrectifiedlinearunit_derivative_test() {
        let activation = LeakyRectifiedLinearUnit::new(0.01f64);
        assert_approx_eq!(activation.derivative(vec![-3.4f64])[0], 0.01f64);
        assert_approx_eq!(activation.derivative(vec![3.4f64])[0], 3.4f64);
    }
}
