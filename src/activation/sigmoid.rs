use activation::Activation;

#[derive(Copy, Clone)]
pub struct Sigmoid;

impl Sigmoid {
    pub fn new() -> Sigmoid {
        return Sigmoid;
    }
}

impl Activation for Sigmoid {
    /// Calculates the Sigmoid of input `x`
    fn calc(&self, x: Vec<f64>) -> Vec<f64> {
        x.iter()
            .map(|n| 1f64 / (1f64 + (-n).exp()))
            .collect::<Vec<_>>()
    }

    /// Calculates the Derivative Sigmoid of input `x`
    fn derivative(&self, x: Vec<f64>) -> Vec<f64> {
        x.iter().map(|n| n * (1f64 - n)).collect::<Vec<_>>()
    }
}

#[cfg(test)]
mod tests {
    use super::Activation;
    use super::Sigmoid;

    #[test]
    fn sigmoid_test() {
        let activation = Sigmoid::new();
        assert_approx_eq!(activation.calc(vec![5f64])[0], 0.9933071490f64);
    }

    #[test]
    fn sigmoid_derivative_test() {
        let activation = Sigmoid::new();
        assert_approx_eq!(activation.derivative(vec![5f64])[0], -20f64);
    }

    #[test]
    fn sigmoid_derivative_correctness_test() {
        let activation = Sigmoid::new();
        let delta = 1e-10f64;
        let val = vec![0.5f64, 0.1f64, 0.9f64];
        let val_delta = val.iter().map(|n| n + delta).collect::<Vec<_>>();

        let approx = activation
            .calc(val_delta)
            .iter()
            .zip(activation.calc(val.clone()).iter())
            .map(|(n, m)| (n - m) / delta)
            .collect::<Vec<_>>();

        let actual = activation.derivative(activation.calc(val.clone()));

        for (n, m) in approx.iter().zip(actual.iter()) {
            assert_approx_eq!(n, m);
        }
    }
}
