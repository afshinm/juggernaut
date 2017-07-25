use matrix::Matrix;
use matrix::MatrixTrait;
use activation::Activation;

/// Represents a neural layer with its weights

#[derive(Clone)]
pub struct NeuralLayer<T: Activation> {
    pub activation: T,
    pub inputs: usize,
    pub neurons: usize,
    pub weights: Matrix
}

impl<T: Activation> NeuralLayer<T> {
    pub fn new(neurons: usize, inputs: usize, activation: T) -> NeuralLayer<T> {
        NeuralLayer {
            activation: activation,
            inputs: inputs,
            neurons: neurons,
            weights: Matrix::random(inputs, neurons)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use matrix::MatrixTrait;
    use activation::Activation;
    use activation::Sigmoid;

    #[test]
    fn new_neural_layer() {
        let test = NeuralLayer::new(4, 3, Sigmoid::new());
        assert_eq!(3usize, test.inputs);
        assert_eq!(4usize, test.neurons);
        assert_eq!(3usize, test.weights.rows());
        assert_eq!(4usize, test.weights.cols());
    }
}
