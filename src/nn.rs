use nl::NeuralLayer;
use activation::Activation;
use activation::Sigmoid;
use sample::Sample;
use matrix::Matrix;
use matrix::MatrixTrait;
use utils::samples_input_to_matrix;
use utils::samples_output_to_matrix;

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
                self.samples[0].get_inputs_count()
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

        let mut weights: Vec<Matrix> = vec![];

        let mut prev_weight: Matrix = Matrix::zero(0, 0);

        for (i, layer) in self.layers.iter().enumerate() {
            // TODO: this part is ridiculously complicated, needs refactoring.
            // and the reason is Rust's lifetime. clean this part, please.

            if i > 0 {
                let mult: Matrix = prev_weight.dot(&layer.weights).map(&|n| self.activation.calc(n));

                if i != self.layers.len() - 1 {
                    prev_weight = mult.clone();
                }

                weights.push(mult);

            } else {
                // first layer (first iteration)
                let samples_input: Matrix = samples_input_to_matrix(&samples);

                let mult: Matrix = samples_input.dot(&layer.weights).map(&|n| self.activation.calc(n));

                if self.layers.len() > 1 {
                    // more than one layer
                    // storing the result for the next iteration
                    prev_weight = mult.clone();
                }

                weights.push(mult);
            }
        }

        weights
    }

    pub fn train(&mut self, epochs: i32) {
        for _ in 0..epochs {
            let mut output: Vec<Matrix> = self.forward(&self.samples);

            // because we are backpropagating
            output.reverse();

            let mut error: Matrix = Matrix::zero(0, 0);
            let mut delta: Matrix = Matrix::zero(0, 0);

            for (i, layer) in output.iter().enumerate() {
                // because it is different when we want to calculate error for each layer for the
                // output layer it is:
                //
                //      y - output_layer
                //
                // but for other layers it is:
                //
                //      output_delta.dot(weights_1)
                //
                if i == 0 {
                    //last layer (output)
                    let samples_outputs: Matrix = samples_output_to_matrix(&self.samples);

                    // this is:
                    //
                    //     y - last_layer_of_forward
                    //
                    // where `last_layer_of_forward` is `layer` because of i == 0 condition
                    //
                    error = Matrix::generate(
                        samples_outputs.rows(),
                        samples_outputs.cols(),
                        &|m,n| samples_outputs.get(m,n) - layer.get(m,n)
                    );
                } else {
                    // this is:
                    //
                    //     delta_of_previous_layer.dot(layer)
                    //
                    error = delta.transpose().dot(&layer);
                }

                let forward_derivative: Matrix = layer.map(&|n| self.activation.derivative(n));
                delta = Matrix::generate(layer.rows(), layer.cols(), &|m,n| layer.get(m, n) * forward_derivative.get(m, n));

                let mut prev_layer: Matrix = samples_input_to_matrix(&self.samples);

                if i != output.len() - 1 {
                    // TODO (afshinm): is this necessary to clone here?
                    prev_layer = output[output.len() - i - 1].clone();
                }

                // updating weights of this layer
                let syn: Matrix = prev_layer.transpose().dot(&delta);

                let index: usize = self.layers.len() - 1 - i;
                // forward output and network layers are the same, with a reversed order
                // TODO (afshinm): is this necessary to clone here?
                let this_layer_weights: Matrix = self.layers[index].weights.clone();

                // finally, set the new weights
                self.layers[index].weights = Matrix::generate(
                    this_layer_weights.rows(),
                    this_layer_weights.cols(),
                    &|m,n| syn.get(m, n) + this_layer_weights.get(m, n)
                );
            }
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
    use matrix::Matrix;
    use matrix::MatrixTrait;

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

        //assert_eq!(test.get_inputs_count(), 2usize);
        //assert_eq!(test.get_outputs_count(), 1usize);
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

        assert_eq!(forward.len(), 2);
    }

    #[test]
    fn train_test() {
        let dataset = vec![Sample::new(vec![1f64, 0f64], vec![0f64])];

        let mut test = NeuralNetwork::new(dataset, Sigmoid::new());

        // 1st layer = 1 neurons - 2 inputs
        test.add_layer(NeuralLayer::new(1, 2));

        let forward = test.forward(&test.samples);

        test.train(10);

        assert_eq!(forward.len(), 1);
    }

    #[test]
    fn train_test_2layers() {
        let dataset = vec![
            Sample::new(vec![1f64, 0f64], vec![0f64]),
            Sample::new(vec![1f64, 1f64], vec![1f64])
        ];

        let mut test = NeuralNetwork::new(dataset, Sigmoid::new());

        // 1st layer = 3 neurons - 2 inputs
        test.add_layer(NeuralLayer::new(3, 2));
        // 2nd layer = 1 neuron - 3 inputs
        test.add_layer(NeuralLayer::new(1, 3));

        let forward = test.forward(&test.samples);

        test.train(5);

        assert_eq!(forward.len(), 2);
    }
}
