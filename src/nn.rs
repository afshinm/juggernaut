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

    pub fn get_inputs_count(&self) -> usize {
        self.samples[0].inputs.len()
    }

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

    #[test]
    fn new_neural_network() {
        let test = NeuralNetwork::new(vec![Sample::new(vec![1f64, 0f64], vec![0f64])], Sigmoid::new());
    }
}
