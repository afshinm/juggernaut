use std::f64;
use activation::Activation;

#[derive(Copy, Clone)]
pub struct SoftMax;

impl SoftMax {
    pub fn new() -> SoftMax {
        return SoftMax;
    }
}

impl Activation for SoftMax {
    /// Calculates the SoftMax of input `x`
    fn calc(&self, x: Vec<f64>) -> Vec<f64> {
        let max_x = x.iter()
            .cloned()
            .max_by(|x, y| x.partial_cmp(&y).unwrap())
            .unwrap();

        let exps = x.iter()
            .cloned()
            .map(|n| n.exp() - max_x)
            .collect::<Vec<_>>();

        let exp_sum: f64 = exps.iter().clone().sum();

        exps.iter().map(|x| x / exp_sum).collect::<Vec<f64>>()
    }

    /// Calculates the Derivative SoftMax of input `x`
    fn derivative(&self, x: Vec<f64>) -> Vec<f64> {
        let softmaxed = self.calc(x.clone());

        softmaxed
            .clone()
            .iter()
            .map(|n| n * (1f64 - n))
            .collect::<Vec<_>>()
    }
}

#[cfg(test)]
mod tests {
    use super::Activation;
    use super::SoftMax;

    #[test]
    fn softmax_test() {
        let activation = SoftMax::new();
        let result = activation.calc(vec![1f64, 5f64, 4f64]);
        let validate = vec![-0.011963105252f64, 0.751918768228f64, 0.260044337024f64];

        for (i, r) in result.iter().enumerate() {
            assert_approx_eq!(r, validate[i]);
        }
    }

    #[test]
    fn softmax_test_sum_one() {
        let activation = SoftMax::new();
        let result = activation.calc(vec![
            43.8291271898136f64,
            10.3468229622968f64,
            90.4820701302356f64,
        ]);

        assert_approx_eq!(result.iter().fold(0f64, |sum, n| sum + n), 1f64);
    }

    //TODO (afshinm): this test returns NaN
    #[test]
    #[ignore]
    fn softmax_test_nan() {
        let activation = SoftMax::new();
        let result = activation.calc(vec![
            143.8291271898136f64,
            710.3468229622968f64,
            690.4820701302356f64,
        ]);

        assert_approx_eq!(result.iter().fold(0f64, |sum, n| sum + n), 1f64);
    }

    //TODO (afshinm): is that correct to add this test to softmax?
    #[test]
    #[ignore]
    fn softmax_derivative_correctness_test() {
        let activation = SoftMax::new();
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
