use nl::NeuralLayer;
use activation::Activation;
use activation::Sigmoid;
use sample::Sample;

/// Represents a Neural Network with layers, inputs and outputs
pub struct NeuralNetwork<T: Activation> {
    activation: T,
    layers: Vec<NeuralLayer>,
    samples: Vec<Sample>
}

impl<T: Activation> NeuralNetwork<T> {
    pub fn new(samples: Vec<Sample>, activation: T) -> NeuralNetwork<T>
        where T: Activation
    {
        let initial_layers: Vec<NeuralLayer> = vec![];

        NeuralNetwork {
            activation: activation,
            layers: initial_layers,
            samples: samples
        }
    }

    /// Returns the number of inputs for one Sample object
    ///
    /// Example:
    ///
    /// ```
    /// # #[macro_use] extern crate juggernaut;
    /// # fn main() {
    /// use juggernaut::sample::Sample;
    /// use juggernaut::nl::NeuralLayer;
    /// use juggernaut::nn::NeuralNetwork;
    /// use juggernaut::activation::Activation;
    /// use juggernaut::activation::Sigmoid;
    ///
    /// let dataset = vec![Sample::new(vec![1f64, 0f64], vec![0f64])];
    /// let mut test = NeuralNetwork::new(dataset, Sigmoid::new());
    ///
    /// assert_eq!(test.get_inputs_count(), 2usize);
    /// # }
    /// ```
    pub fn get_inputs_count(&self) -> usize {
        self.samples[0].inputs.len()
    }

    /// To add a new layer to the network
    ///
    /// Example:
    ///
    /// ```
    /// # #[macro_use] extern crate juggernaut;
    /// # fn main() {
    /// use juggernaut::sample::Sample;
    /// use juggernaut::nl::NeuralLayer;
    /// use juggernaut::nn::NeuralNetwork;
    /// use juggernaut::activation::Activation;
    /// use juggernaut::activation::Sigmoid;
    ///
    /// let dataset = vec![Sample::new(vec![1f64, 0f64], vec![0f64])];
    /// let mut test = NeuralNetwork::new(dataset, Sigmoid::new());
    ///
    /// // 1st layer = 4 neurons - 2 inputs
    /// let nl1 = NeuralLayer::new(4, 2);
    ///
    /// test.add_layer(nl1);
    /// # }
    /// ```
    pub fn add_layer(&mut self, layer: NeuralLayer) {

        let prev_layer_neurons: usize = match self.layers.last() {
            Some(l) => l.neurons,
            None => self.get_inputs_count() // no hidden layer
        };

        if prev_layer_neurons != layer.inputs {
            panic!("New layer should have enough inputs. \
                   Expected {}, got {}", prev_layer_neurons, layer.inputs);
        }

        self.layers.push(layer);
    }

    pub fn train(&self) {

    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use activation::Sigmoid;
    use activation::Activation;
    use sample::Sample;
    use nl::NeuralLayer;

    #[test]
    fn new_neural_network() {
        let dataset = vec![Sample::new(vec![1f64, 0f64], vec![0f64])];

        let mut test = NeuralNetwork::new(dataset, Sigmoid::new());

        // 1st layer = 4 neurons - 2 inputs
        let nl1 = NeuralLayer::new(4, 2);
        // 2nd layer = 3 neurons - 4 inputs
        let nl2 = NeuralLayer::new(3, 4);

        test.add_layer(nl1);
        test.add_layer(nl2);

    }
}
