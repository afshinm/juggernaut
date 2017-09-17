use matrix::Matrix;
use matrix::MatrixTrait;
use activation::Activation;

/// Represents a neural layer with its weights
pub struct NeuralLayer {
    pub activation: Box<Activation>,
    inputs: usize,
    neurons: usize,
    weights: Matrix,
    biases: Matrix,
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
            weights: Matrix::random(neurons, inputs),
            biases: Matrix::random(neurons, 1),
        }
    }

    pub fn neurons(&self) -> usize {
        self.neurons
    }

    pub fn inputs(&self) -> usize {
        self.inputs
    }

    pub fn biases(&self) -> &Matrix {
        &self.biases
    }

    // weights without bias node
    pub fn weights(&self) -> &Matrix {
        &self.weights
    }

    // weights with bias node
    pub fn weights_with_bias(&self) -> &Matrix {
        &self.weights
    }

    pub fn set_weights(&mut self, weights: Matrix) {
        // because no one can change the dimension of the matrix
        assert!(weights.rows() == self.weights.rows());
        assert!(weights.cols() == self.weights.cols());

        self.weights = weights;
    }

    pub fn set_biases(&mut self, weights: Matrix) {
        // because no one can change the dimension of the matrix
        assert!(weights.rows() == self.biases.rows());
        assert!(weights.cols() == self.biases.cols());

        self.biases = weights;
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
        assert_eq!(3usize, test.inputs());
        assert_eq!(4usize, test.neurons());

        assert_eq!(4usize, test.weights().rows());
        assert_eq!(3usize, test.weights().cols());
    }

    /*
    #[test]
    fn neural_layer_bias() {
        let test = NeuralLayer::new(4, 3, Sigmoid::new());

        assert_eq!(4usize, test.weights_with_bias().rows());
        // 4 because of bias node
        assert_eq!(4usize, test.weights_with_bias().cols());
    }
    */
}
