pub mod sigmoid;
pub mod identity;
pub mod hyperbolictangent;
pub mod softplus;
pub mod rectifiedlinearunit;
pub mod leakyrectifiedlinearunit;

pub use self::sigmoid::Sigmoid;
pub use self::identity::Identity;
pub use self::hyperbolictangent::HyperbolicTangent;
pub use self::softplus::SoftPlus;
pub use self::rectifiedlinearunit::RectifiedLinearUnit;
pub use self::leakyrectifiedlinearunit::LeakyRectifiedLinearUnit;

/// Activation functions
pub trait Activation {
    // the function itself
    fn calc(&self, x: f64) -> f64;
    // Derivative
    fn derivative(&self, x: f64) -> f64;
}
