use matrix::Matrix;
use matrix::MatrixTrait;

/// Represents a neural layer with its weights
pub struct NeuralLayer {
    inputs: usize,
    neurons: usize,
    weights: Matrix
}

impl NeuralLayer {
    pub fn new(neurons: usize, inputs: usize) -> NeuralLayer {
        NeuralLayer {
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

    #[test]
    fn new_neural_layer() {
        let test = NeuralLayer::new(4, 3);
        assert_eq!(3usize, test.inputs);
        assert_eq!(4usize, test.neurons);
        assert_eq!(3usize, test.weights.rows());
        assert_eq!(4usize, test.weights.cols());
    }
}
