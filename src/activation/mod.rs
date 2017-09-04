pub mod sigmoid;
pub mod identity;
pub mod hyperbolictangent;
pub mod softplus;
pub mod softmax;
pub mod rectifiedlinearunit;
pub mod leakyrectifiedlinearunit;

pub use self::sigmoid::Sigmoid;
pub use self::identity::Identity;
pub use self::hyperbolictangent::HyperbolicTangent;
pub use self::softplus::SoftPlus;
pub use self::softmax::SoftMax;
pub use self::rectifiedlinearunit::RectifiedLinearUnit;
pub use self::leakyrectifiedlinearunit::LeakyRectifiedLinearUnit;

/// Activation trait
pub trait Activation {
    // the function itself
    fn calc(&self, x: f64) -> f64;
    // Derivative
    fn derivative(&self, x: f64) -> f64;
}

/// Activation trait for Vec input
// TODO (afshin): We need to merge these two traits. 
// This trait is only used for Softmax
pub trait ActivationVec {
    // the function itself
    fn calc(&self, x: Vec<f64>) -> Vec<f64>;
    // Derivative
    fn derivative(&self, x: Vec<f64>) -> Vec<f64>;
}
