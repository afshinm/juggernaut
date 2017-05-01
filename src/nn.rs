use nl::NeuralLayer;
use activation::Activation;
use activation::Sigmoid;

/// Represents a Neural Network with layers, inputs and outputs
pub struct NeuralNetwork<T: Activation> {
    activation: T,
    layers: Vec<NeuralLayer>
}

impl<T: Activation> NeuralNetwork<T> {
    pub fn new(inputs: Vec<f64>,
               outputs: Vec<f64>,
               activation: T) -> NeuralNetwork<T>
        where T: Activation
    {
        let initial_layers: Vec<NeuralLayer> = vec![];

        NeuralNetwork {
            activation: activation,
            layers: initial_layers
        }
    }

    pub fn add_hidden_layer(&mut self, layer: NeuralLayer) {
        self.layers.push(layer);
    }

    pub fn train(&self) {

    }
}
