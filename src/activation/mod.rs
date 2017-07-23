pub mod sigmoid;
pub mod identity;
pub mod hyperbolictangent;
pub mod softplus;
pub mod rectifiedlinearunit;

pub use self::sigmoid::Sigmoid;
pub use self::identity::Identity;
pub use self::hyperbolictangent::HyperbolicTangent;
pub use self::softplus::SoftPlus;
pub use self::rectifiedlinearunit::RectifiedLinearUnit;

/// Activation functions
pub trait Activation {
    fn new() -> Self;
    // the function itself
    fn calc(&self, x: f64) -> f64;
    // Derivative
    fn derivative(&self, x: f64) -> f64;
}


