use nl::NeuralLayer;
use activation::Activation;
use activation::Sigmoid;
use sample::Sample;
use matrix::Matrix;
use matrix::MatrixTrait;
use utils::samples_input_to_matrix;

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
            let mut output: Vec<Matrix> = self.forward(&self.samples);
            output.reverse();

            for (i, layer) in output.iter().enumerate() {
                println!("one {:?}", layer);

                // because it is different when we want to calculate error for each layer for the
                // output layer it is:
                //
                //      y - output_layer
                //
                // but for other layers it is:
                //
                //      output_delta.dot(weights_1)
                //
                if (i == 0) {
                    //last layer (output)
                    let error: Matrix = self.output_delta(&self.samples, &output);
                } else {

                }
            }

            let mut output_derivative: Vec<Matrix> = vec![];
            let mut derivative_error: Vec<f64> = vec![];


            //println!("error {:?}", error);

            // TODO (afshinm): changing the forward output to Matrix (from Vec<Matrix>) removes
            // this loop. e.g. `sigmoid_derivative(output)`
            for this_output in output {
                 output_derivative.push(this_output.map(&|n| self.activation.derivative(n)));
            }

            for (i, derivative) in output_derivative.iter().enumerate() {
                // TODO (afshinm): what if the output is more then one?
                // should we use matrix.dot?
                derivative_error.push(derivative.get(0,0) * error[i].get(0,0));
            }

            let matrix_of_inputs: Matrix = samples_input_to_matrix(&self.samples);
            let matrix_of_derivative: Matrix = Matrix::from_vec(&derivative_error);

            //println!("output_derivative {:?}", output_derivative);
            //println!("error * der {:?}", derivative_error);
            println!("samples {:?}", matrix_of_inputs);
            println!("derivative error {:?}", matrix_of_derivative);

            let adjustment: Matrix = matrix_of_derivative.dot(&matrix_of_inputs);

            println!("adjustment {:?}", adjustment);
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

    /*

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
    fn output_delta_test() {
        let dataset = vec![Sample::new(vec![1f64, 0f64], vec![1f64])];

        let mut test = NeuralNetwork::new(dataset, Sigmoid::new());

        let mock_output = vec![Matrix::zero(1, 1)];

        let output_delta = test.output_delta(&test.samples, &mock_output);

        assert_eq!(output_delta.len(), 1);
        assert_eq!(output_delta[0].get(0,0), 1f64);
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

    */

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

        test.train(1);

        assert_eq!(forward.len(), 2);
    }
}
