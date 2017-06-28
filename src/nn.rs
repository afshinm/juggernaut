use nl::NeuralLayer;
use activation::Activation;
use activation::Sigmoid;
use sample::Sample;
use matrix::Matrix;
use matrix::MatrixTrait;

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
        let mut initial_layers: Vec<NeuralLayer> = vec![];

        // adding the first layer, which is a layer that connects inputs to outputs
        //
        // TODO: I commented this line because we have to let user to decide about the layers. do
        // we need a default layer when user doesn't define the layers?
        //initial_layers.push(NeuralLayer::new(samples[0].get_outputs_count(), samples[0].get_inputs_count()));

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
        self.samples[0].get_inputs_count()
    }

    /// Returns the number of outputs for one Sample object
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
    /// assert_eq!(test.get_outputs_count(), 1usize);
    /// # }
    /// ```
    pub fn get_outputs_count(&self) -> usize {
        self.samples[0].get_outputs_count()
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
        let mut layers = self.layers.to_owned();

        let prev_layer_neurons: usize = {
            if layers.len() > 0 {
                // 1 for len()
                layers[layers.len() - 1].neurons
            } else {
                self.get_inputs_count()
            }
        };

        if prev_layer_neurons != layer.inputs {
            panic!("New layer should have enough inputs. \
                   Expected {}, got {}", prev_layer_neurons, layer.inputs);
        }

        self.layers.push(layer);
    }

    /// This is the forward method of the network which calculates the random weights
    /// and multiplies the inputs of given samples to the weights matrix. Thinks.
    pub fn forward(&self, samples: &Vec<Sample>) -> Vec<Matrix> {
        if self.layers.len() == 0 {
            panic!("Neural network doesn't have any layers.");
        }

        // TODO (afshinm): is this correct to store weights in a vector of Matrix?
        // another idea: storing as a vector of f64s
        let mut weights: Vec<Matrix> = vec![];

        for sample in samples.iter() {
            let mut prev_weight: Matrix = Matrix::zero(0, 0);

            for (i, layer) in self.layers.iter().enumerate() {
                // TODO: this part is ridiculously complicated, needs refactoring.
                // and the reason is Rust's lifetime. clean this part, please.

                if i > 0 {
                    if i == self.layers.len() - 1 {
                        // last iteration
                        weights.push(prev_weight.dot(&layer.weights));
                    } else {
                        prev_weight = prev_weight.dot(&layer.weights);
                    }

                } else {
                    // first layer (first iteration)
                    let mut first: Matrix = Matrix::from_vec(&sample.inputs);

                    if self.layers.len() == 1 {
                        // does the network have only one layer?
                        weights.push(first.dot(&layer.weights));
                    } else {
                        // more than one layer
                        // storing the result for the next iteration
                        prev_weight = first.dot(&layer.weights);
                    }
                }
            }
        }

        weights
    }

    /// Caculates the delta of `forward` step with given samples
    /// Used for training step
    fn output_delta(&self, samples: &Vec<Sample>, forward_output: &Vec<Matrix>) -> Vec<Matrix> {
        let mut delta: Vec<Matrix> = vec![];

        // assumed that samples and forward_output is in the same order
        for (i, sample) in samples.iter().enumerate() {
            // single output of forward pass
            let this_forward_output: &Matrix = &forward_output[i];

            // TODO (afshinm): is this correct to store the delta in a vector
            // and then covert it to a Matrix? or maybe we should use Matrix and push elements.
            let mut this_delta: Vec<f64> = vec![];

            for (j, output) in sample.outputs.iter().enumerate() {
                this_delta.push(output - this_forward_output.get(0, j));
            }

            delta.push(Matrix::from_vec(&this_delta));
        }

        delta
    }

    pub fn train(&self, epochs: i32) {
        for _ in 0..epochs {
            let output: Vec<Matrix> = self.forward(&self.samples);
            let delta = self.output_delta(&self.samples, &output);
        }
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
    fn new_neural_network_test() {
        let dataset = vec![Sample::new(vec![1f64, 0f64], vec![0f64])];

        let mut test = NeuralNetwork::new(dataset, Sigmoid::new());

        // 1st layer = 4 neurons - 2 inputs
        let nl1 = NeuralLayer::new(4, 2);
        // 2nd layer = 3 neurons - 4 inputs
        let nl2 = NeuralLayer::new(3, 4);

        test.add_layer(nl1);
        test.add_layer(nl2);

        assert_eq!(test.get_inputs_count(), 2usize);
        assert_eq!(test.get_outputs_count(), 1usize);
    }

    #[test]
    fn forward_test() {
        let dataset = vec![Sample::new(vec![1f64, 0f64], vec![0f64])];

        let mut test = NeuralNetwork::new(dataset, Sigmoid::new());

        // 1st layer = 1 neurons - 2 inputs
        test.add_layer(NeuralLayer::new(1, 2));

        let forward = test.forward(&test.samples);
        assert_eq!(forward.len(), 1);
    }

    #[test]
    fn forward_test_2layers() {
        let dataset = vec![Sample::new(vec![1f64, 0f64], vec![0f64])];

        let mut test = NeuralNetwork::new(dataset, Sigmoid::new());

        // 1st layer = 3 neurons - 2 inputs
        test.add_layer(NeuralLayer::new(3, 2));
        // 2nd layer = 1 neuron - 3 inputs
        test.add_layer(NeuralLayer::new(1, 3));

        let forward = test.forward(&test.samples);

        assert_eq!(forward.len(), 1);
    }

    #[test]
    fn train_test() {
        let dataset = vec![Sample::new(vec![1f64, 0f64], vec![0f64])];

        let mut test = NeuralNetwork::new(dataset, Sigmoid::new());

        // 1st layer = 1 neurons - 2 inputs
        test.add_layer(NeuralLayer::new(1, 2));

        let forward = test.forward(&test.samples);

        test.train(1000);

        assert_eq!(forward.len(), 1);
    }
}
