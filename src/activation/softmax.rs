use std::f64;
use activation::ActivationVec;

#[derive(Copy, Clone)]
pub struct SoftMax;

impl SoftMax {
    pub fn new() -> SoftMax {
        return SoftMax;
    }
}

impl ActivationVec for SoftMax {
    /// Calculates the SoftMax of input `x`
    fn calc(&self, x: Vec<f64>) -> Vec<f64> {
        let exps = x.iter().map(|x| x.exp());
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
    use super::ActivationVec;
    use super::SoftMax;

    #[test]
    fn softmax_test() {
        let activation = SoftMax::new();
        let result = activation.calc(vec![1f64, 5f64, 4f64]);
        let validate = vec![1f64, 1f64, 1f64];

        for (i, r) in result.iter().enumerate() {
            assert_approx_eq!(r, validate[i]);
        }
    }
}
