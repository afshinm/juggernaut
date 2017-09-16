use matrix::Matrix;
use matrix::MatrixTrait;
use activation::Activation;

/// Represents a neural layer with its weights
pub struct NeuralLayer {
    pub activation: Box<Activation>,
    inputs: usize,
    neurons: usize,
    weights: Matrix,
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
        }
    }

    pub fn neurons(&self) -> usize {
        self.neurons
    }

    pub fn inputs(&self) -> usize {
        self.inputs
    }

    // weights with bias node
    pub fn weights(&self) -> &Matrix {
        &self.weights
    }

    pub fn set_weights(&mut self, weights: Matrix) {
        self.weights = weights;
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
        
        assert_eq!(4usize, test.weights.rows());
        assert_eq!(3usize, test.weights.cols());
    }
}
