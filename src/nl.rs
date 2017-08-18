use matrix::Matrix;
use matrix::MatrixTrait;
use activation::Activation;

/// Represents a neural layer with its weights
pub struct NeuralLayer {
    pub activation: Box<Activation>,
    pub inputs: usize,
    pub neurons: usize,
    pub weights: Matrix,
}

impl NeuralLayer {
    pub fn new<T: 'static>(neurons: usize, inputs: usize, activation: T) -> NeuralLayer
    where
        T: Activation,
    {
        NeuralLayer {
            activation: Box::new(activation),
            inputs: inputs,
            neurons: neurons,
            weights: Matrix::random(inputs, neurons),
        }
    }
}

#[cfg(test)]
mod tests {
    use nl::NeuralLayer;
    use matrix::MatrixTrait;
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
