pub mod squared_error;
pub mod cross_entropy;

use matrix::Matrix;

/// Available cost functions
/// The only reason for having this enum is to `match` it in `NeuralNetwork`
pub enum CostFunctions {
    SquaredError,
    CrossEntropy,
}

/// Trait of cost functions
pub trait CostFunction {
    // calculates the value of cost function
    fn calc(&self, prediction: &Matrix, target: &Matrix) -> f64;
    // returns the corresponding enum
    // TODO (afshinm): the only usage of this method is for `match`ing in NeuralNetwork 
    // can we find a better way to do this?
    fn name(&self) -> CostFunctions;
}
