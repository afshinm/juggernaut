pub mod squared_error;

use matrix::Matrix;

/// Available cost functions
/// The only reason for having this enum is to `match` it in `NeuralNetwork`
pub enum CostFunctions {
    SquaredError,
    CrossEntropy
}

/// Trait of cost functions
pub trait CostFunction {
    // calculates the value of 
    fn calc(&self, prediction: &Matrix, target: &Matrix) -> f64;
    // returns the corresponding enum
    fn name(&self) -> CostFunctions;
}
