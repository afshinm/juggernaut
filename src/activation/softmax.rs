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
        let exps = x.iter().cloned().map(f64::exp as fn(f64) -> f64);
        let exp_sum: f64 = exps.clone().sum();
        exps.map(|x| x / exp_sum).collect::<Vec<f64>>()
    }

    /// Calculates the Derivative SoftMax of input `x`
    fn derivative(&self, x: Vec<f64>) -> Vec<f64> {
        vec![]
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
        let validate = vec![0.013212886f64, 0.72139918427f64, 0.2653879287f64];

        for (i, r) in result.iter().enumerate() {
            assert_approx_eq!(r, validate[i]);
        }
    }
}
