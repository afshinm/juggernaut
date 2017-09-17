use nl::NeuralLayer;
use sample::Sample;
use matrix::Matrix;
use matrix::MatrixTrait;
use cost::CostFunction;
use cost::squared_error::SquaredError;
use cost::CostFunctions;
use utils::samples_input_to_matrix;
use utils::sample_input_to_matrix;
use utils::samples_output_to_matrix;
use utils::sample_output_to_matrix;

/// Represents a Neural Network with layers, inputs and outputs
pub struct NeuralNetwork {
    layers: Vec<NeuralLayer>,
    cost_function: Box<CostFunction>,
    on_error_fn: Option<Box<Fn(f64)>>,
    on_epoch_fn: Option<Box<Fn(&NeuralNetwork)>>,
}

impl NeuralNetwork {
    pub fn new() -> NeuralNetwork {
        NeuralNetwork {
            layers: vec![],
            cost_function: Box::new(SquaredError::new()),
            on_error_fn: None,
            on_epoch_fn: None,
        }
    }

    /// To set a cost function for the network
    pub fn set_cost_function<T>(&mut self, cost_function: T)
    where
        T: 'static + CostFunction,
    {
        self.cost_function = Box::new(cost_function);
    }

    /// To add a callback function and receive the errors of the network during training process
    /// Please note that there is another function that basically calcualtes the error value
    pub fn on_error<FN>(&mut self, callback_fn: FN)
    where
        FN: 'static + Fn(f64),
    {
        self.on_error_fn = Some(Box::new(callback_fn));
    }

    /// To add a callback function to get called after each epoch
    pub fn on_epoch<FN>(&mut self, callback_fn: FN)
    where
        FN: 'static + Fn(&NeuralNetwork),
    {
        self.on_epoch_fn = Some(Box::new(callback_fn));
    }

    /// To emit the `on_error` callback
    fn emit_on_error(&self, err: f64) {
        match self.on_error_fn {
            Some(ref err_fn) => err_fn(err),
            None => (),
        }
    }

    /// To emit the `on_epoch` callback
    fn emit_on_epoch(&self) {
        match self.on_epoch_fn {
            Some(ref epoch_fn) => epoch_fn(&self),
            None => (),
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
    /// let mut test = NeuralNetwork::new();
    ///
    /// // 1st layer = 4 neurons - 2 inputs
    /// let nl1 = NeuralLayer::new(4, 2, Sigmoid::new());
    ///
    /// test.add_layer(nl1);
    /// # }
    /// ```
    pub fn add_layer(&mut self, layer: NeuralLayer) {
        if self.layers.len() > 0 {
            let prev_layer_neurons = self.layers[self.layers.len() - 1].neurons();

            if prev_layer_neurons != layer.inputs() {
                panic!(
                    "New layer should have enough inputs. \
                     Expected {}, got {}",
                    prev_layer_neurons,
                    layer.inputs()
                );
            }
        }

        self.layers.push(layer);
    }

    /// To get the layers of the network
    pub fn get_layers(&self) -> &Vec<NeuralLayer> {
        &self.layers
    }

    /// This helper function is used in `forward` step to apply the filter function of cost
    /// function.
    /// For instance, we need to apply a Softmax for CrossEntropy
    fn apply_cost_function_filter(&self, input: Matrix) -> Matrix {
        match (*self.cost_function).name() {
            CostFunctions::CrossEntropy => {
                use activation::SoftMax;
                use activation::Activation;

                input.map_row(&|row| SoftMax::new().calc(row))
            }
            CostFunctions::SquaredError => input,
            _ => panic!("Invalid cost function."),
        }
    }

    /// This is the forward method of the network which calculates the random weights
    /// and multiplies the inputs of given samples to the weights matrix. Thinks.
    pub fn forward(&self, sample: &Sample) -> Vec<Matrix> {
        if self.layers.len() == 0 {
            panic!("Neural network doesn't have any layers.");
        }

        let mut weights: Vec<Matrix> = vec![];

        let mut prev_weight: Matrix = Matrix::zero(0, 0);

        for (i, layer) in self.layers.iter().enumerate() {
            // TODO: this part is ridiculously complicated, needs refactoring.
            // and the reason is Rust's lifetime. clean this part, please.
            //
            let transposed_bias = layer.biases().transpose();

            if i > 0 {
                // TODO (afshinm): can we use a map_row to take advantage of activation(Vec<f64>)?
                let mut mult: Matrix = prev_weight.dot(&layer.weights().transpose()).map(&|n, i, j| {
                    *layer.activation.calc(vec![n +  (1f64 * transposed_bias.get(0, j))]).last().unwrap()
                });

                if i != self.layers.len() - 1 {
                    prev_weight = mult.clone();
                } else {
                    // this is the last layer (output)
                    mult = self.apply_cost_function_filter(mult);
                }

                weights.push(mult);

            } else {
                // first layer (first iteration)
                let samples_input: Matrix = sample_input_to_matrix(&sample);

                let mut mult: Matrix = samples_input.dot(&layer.weights().transpose()).map(&|n, i, j| {
                    *layer.activation.calc(vec![n +  (1f64 * transposed_bias.get(0, j))]).last().unwrap()
                });

                if self.layers.len() > 1 {
                    // more than one layer
                    // storing the result for the next iteration
                    prev_weight = mult.clone();
                } else {
                    // this is the last layer (output)
                    mult = self.apply_cost_function_filter(mult);
                }

                weights.push(mult);
            }
        }

        weights
    }

    /// Use this function to evaluate a trained neural network
    ///
    /// This function simply passes the given sample to the `forward` function and returns the
    /// output of last layer
    pub fn evaluate(&self, sample: Sample) -> Matrix {
        let forward: Vec<Matrix> = self.forward(&sample);

        // TODO (afshinm): is this correct to clone here?
        forward.last().unwrap().clone()
    }

    /// This function calculates the error rate of network during training and
    /// calls the `on_error_fn` if it is available
    fn error(&self, prediction: &Matrix, target: &Matrix) -> f64 {
        let err = self.cost_function.calc(prediction, target);

        self.emit_on_error(err);

        err
    }

    pub fn train(&mut self, samples: Vec<Sample>, epochs: i32, learning_rate: f64) {
        for _ in 0..epochs {

            for (k, sample) in samples.iter().enumerate() {

                let mut output: Vec<Matrix> = self.forward(&sample);

                // because we are backpropagating
                output.reverse();

                //let mut error: Matrix = Matrix::zero(0, 0);
                let mut delta: Matrix = Matrix::zero(0, 0);

                for (i, layer) in output.iter().enumerate() {
                    // because of `reverse`
                    let index: usize = self.layers.len() - 1 - i;

                    // because it is different when we want to calculate error for each layer for the
                    // output layer it is:
                    //
                    //      y - output_layer
                    //
                    // but for other layers it is:
                    //
                    //      output_delta.dot(weights_1)
                    //
                    let error = if i == 0 {
                        //last layer (output)
                        let samples_outputs = sample_output_to_matrix(&sample);

                        // this is:
                        //
                        //     y - last_layer_of_forward
                        //
                        // where `last_layer_of_forward` is `layer` because of i == 0 condition

                        let error =
                            Matrix::generate(samples_outputs.rows(), samples_outputs.cols(), &|m, n| {
                                samples_outputs.get(m, n) - layer.get(m, n)
                            });

                        if samples.len() - 1 == k {
                            // calculating error of this iteration
                            // and call the error_fn to notify
                            self.error(&layer, &samples_outputs);
                        }

                        error
                    } else {
                        // this is:
                        //
                        //     delta_of_previous_layer.dot(layer)
                        //
                        delta.dot(&self.layers[index + 1].weights().clone())
                    };

                    let forward_derivative: Matrix = layer.map(&|n, _, _| {
                        *self.layers[index]
                            .activation
                            .derivative(vec![n])
                            .last()
                            .unwrap()
                    });

                    delta = Matrix::generate(layer.rows(), layer.cols(), &|m, n| {
                        error.get(m, n) * forward_derivative.get(m, n)
                    });

                    let biases = self.layers[index].biases().clone();

                    self.layers[index].set_biases(biases.map(&|n, i, j| {
                        n + (delta.get(j, i) * learning_rate)
                    }));

                    let mut prev_layer: Matrix = sample_input_to_matrix(&sample);

                    if i != output.len() - 1 {
                        // TODO (afshinm): is this necessary to clone here?
                        prev_layer = output[i + 1].clone();
                    }

                    // updating weights of this layer
                    let syn: Matrix = delta.map(&|n, _, _| {
                        n * learning_rate
                    }).transpose().dot(&prev_layer);

                    // forward output and network layers are the same, with a reversed order
                    // TODO (afshinm): is this necessary to clone here?
                    let this_layer_weights: &Matrix = &self.layers[index].weights().clone();

                    // finally, set the new weights
                    self.layers[index].set_weights(Matrix::generate(
                            this_layer_weights.rows(),
                            this_layer_weights.cols(),
                            &|m, n| syn.get(m, n) + this_layer_weights.get(m, n),
                            ));
                }
            }

            // call on_epoch callback
            self.emit_on_epoch();
        }
    }
}

#[cfg(test)]
mod tests {
    use activation::Sigmoid;
    use activation::HyperbolicTangent;
    use sample::Sample;
    use nl::NeuralLayer;
    use nn::NeuralNetwork;
    use matrix::MatrixTrait;

    #[test]
    fn get_layers_test() {
        let mut test = NeuralNetwork::new();

        let layers = vec![NeuralLayer::new(1, 2, Sigmoid::new())];

        for layer in layers {
            test.add_layer(layer);
        }

        let get_layers = test.get_layers();

        assert_eq!(get_layers.len(), 1);
    }

    #[test]
    fn forward_test() {
        let dataset = vec![Sample::new(vec![1f64, 0f64], vec![0f64])];

        let mut test = NeuralNetwork::new();

        let sig_activation = Sigmoid::new();
        // 1st layer = 1 neurons - 2 inputs
        test.add_layer(NeuralLayer::new(1, 2, sig_activation));

        let forward = test.forward(&dataset[0]);
        assert_eq!(forward.len(), 1);
    }

    #[test]
    fn forward_test_2layers() {
        let dataset = vec![Sample::new(vec![1f64, 0f64], vec![0f64])];

        let mut test = NeuralNetwork::new();

        let sig_activation = Sigmoid::new();

        // 1st layer = 3 neurons - 2 inputs
        test.add_layer(NeuralLayer::new(3, 2, sig_activation));
        // 2nd layer = 1 neuron - 3 inputs
        test.add_layer(NeuralLayer::new(1, 3, sig_activation));

        let forward = test.forward(&dataset[0]);

        assert_eq!(forward.len(), 2);
    }

    #[test]
    fn train_test() {
        let dataset = vec![Sample::new(vec![1f64, 0f64], vec![0f64])];

        let mut test = NeuralNetwork::new();

        let sig_activation = Sigmoid::new();

        // 1st layer = 1 neurons - 2 inputs
        test.add_layer(NeuralLayer::new(1, 2, sig_activation));

        test.train(dataset, 10, 0.1f64);
    }

    #[test]
    fn train_test_2layers() {
        let dataset = vec![
            Sample::new(vec![1f64, 0f64], vec![0f64]),
            Sample::new(vec![1f64, 1f64], vec![1f64]),
            Sample::new(vec![1f64, 1f64], vec![1f64]),
        ];

        let mut test = NeuralNetwork::new();

        let sig_activation = Sigmoid::new();

        // 1st layer = 3 neurons - 2 inputs
        //test.add_layer(NeuralLayer::new(3, 2, sig_activation));
        // 2nd layer = 1 neuron - 3 inputs
        test.add_layer(NeuralLayer::new(1, 2, sig_activation));

        //let forward = test.forward(&dataset);

        test.train(dataset, 100, 0.1f64);

        //assert_eq!(forward.len(), 2);
    }


    #[test]
    fn train_test_2layers_think() {
        let dataset = vec![
            Sample::new(vec![0f64, 0f64, 1f64], vec![0f64]),
            Sample::new(vec![0f64, 1f64, 1f64], vec![0f64]),
            Sample::new(vec![1f64, 0f64, 1f64], vec![1f64]),
            Sample::new(vec![1f64, 1f64, 1f64], vec![1f64]),
        ];

        let mut test = NeuralNetwork::new();

        let sig_activation = Sigmoid::new();

        // 1st layer = 2 neurons - 3 inputs
        test.add_layer(NeuralLayer::new(2, 3, sig_activation));
        // 2nd layer = 1 neuron - 2 inputs
        test.add_layer(NeuralLayer::new(1, 2, sig_activation));

        test.train(dataset, 5, 0.1f64);

        let think = test.evaluate(Sample::predict(vec![1f64, 0f64, 1f64]));

        assert_eq!(think.rows(), 1);
        assert_eq!(think.cols(), 1);
    }



    #[test]
    fn error_function_test() {
        let dataset = vec![
            Sample::new(vec![0f64, 0f64, 1f64], vec![0f64]),
            Sample::new(vec![0f64, 1f64, 1f64], vec![0f64]),
            Sample::new(vec![1f64, 0f64, 1f64], vec![1f64]),
            Sample::new(vec![1f64, 1f64, 1f64], vec![1f64]),
        ];

        let mut test = NeuralNetwork::new();

        // error should be more than 0
        test.on_error(|err| {
            assert!(err > 0f64);
        });

        let sig_activation = Sigmoid::new();

        // 1st layer = 2 neurons - 3 inputs
        test.add_layer(NeuralLayer::new(2, 3, sig_activation));
        // 2nd layer = 1 neuron - 2 inputs
        test.add_layer(NeuralLayer::new(1, 2, sig_activation));

        test.train(dataset, 5, 0.1f64);
    }

    #[test]
    fn on_epoch_test() {
        let dataset = vec![
            Sample::new(vec![0f64, 0f64, 1f64], vec![0f64]),
            Sample::new(vec![0f64, 1f64, 1f64], vec![0f64]),
            Sample::new(vec![1f64, 0f64, 1f64], vec![1f64]),
            Sample::new(vec![1f64, 1f64, 1f64], vec![1f64]),
        ];

        let mut test = NeuralNetwork::new();

        // TODO (afshinm): this test is not complete.
        // it should count the number of calls of the closure as well
        test.on_epoch(|this| {
            assert_eq!(3, this.layers[0].weights().cols());
            assert_eq!(2, this.layers[0].weights().rows());

            assert_eq!(2, this.layers[1].weights().cols());
            assert_eq!(1, this.layers[1].weights().rows());
        });

        let sig_activation = Sigmoid::new();

        // 1st layer = 2 neurons - 3 inputs
        test.add_layer(NeuralLayer::new(2, 3, sig_activation));
        // 2nd layer = 1 neuron - 2 inputs
        test.add_layer(NeuralLayer::new(1, 2, sig_activation));

        test.train(dataset, 5, 0.1f64);
    }

    #[test]
    fn network_with_two_activations() {
        let dataset = vec![
            Sample::new(vec![0f64, 0f64, 1f64], vec![0f64]),
            Sample::new(vec![0f64, 1f64, 1f64], vec![0f64]),
            Sample::new(vec![1f64, 0f64, 1f64], vec![1f64]),
            Sample::new(vec![1f64, 1f64, 1f64], vec![1f64]),
        ];

        let mut test = NeuralNetwork::new();

        // 1st layer = 2 neurons - 3 inputs
        test.add_layer(NeuralLayer::new(2, 3, Sigmoid::new()));
        // 2nd layer = 1 neuron - 2 inputs
        test.add_layer(NeuralLayer::new(1, 2, HyperbolicTangent::new()));

        test.train(dataset, 5, 0.1f64);

        let think = test.evaluate(Sample::predict(vec![1f64, 0f64, 1f64]));

        assert_eq!(think.rows(), 1);
        assert_eq!(think.cols(), 1);
    }


    #[test]
    fn two_hidden_layers() {
        let dataset = vec![
            Sample::new(vec![0f64, 0f64, 1f64], vec![0f64]),
            Sample::new(vec![0f64, 1f64, 1f64], vec![0f64]),
            Sample::new(vec![1f64, 0f64, 1f64], vec![1f64]),
            Sample::new(vec![1f64, 1f64, 1f64], vec![1f64]),
        ];

        let mut test = NeuralNetwork::new();

        // 1st layer = 2 neurons - 3 inputs
        test.add_layer(NeuralLayer::new(2, 3, Sigmoid::new()));
        // 2nd layer = 4 neurons - 2 inputs
        test.add_layer(NeuralLayer::new(4, 2, Sigmoid::new()));
        // 3rd layer = 1 neuron - 4 inputs
        test.add_layer(NeuralLayer::new(1, 4, Sigmoid::new()));

        test.train(dataset, 1, 0.1f64);

        let think = test.evaluate(Sample::predict(vec![1f64, 0f64, 1f64]));

        assert_eq!(think.rows(), 1);
        assert_eq!(think.cols(), 1);
    }

    #[test]
    fn three_hidden_layers() {
        let dataset = vec![
            Sample::new(vec![0f64, 0f64, 1f64], vec![0f64]),
            Sample::new(vec![0f64, 1f64, 1f64], vec![0f64]),
            Sample::new(vec![1f64, 0f64, 1f64], vec![1f64]),
            Sample::new(vec![1f64, 1f64, 1f64], vec![1f64]),
        ];

        let mut test = NeuralNetwork::new();

        // 1st layer = 2 neurons - 3 inputs
        test.add_layer(NeuralLayer::new(2, 3, Sigmoid::new()));
        // 2nd layer = 4 neurons - 2 inputs
        test.add_layer(NeuralLayer::new(4, 2, Sigmoid::new()));
        // 3rd layer = 8 neurons - 4 inputs
        test.add_layer(NeuralLayer::new(8, 4, Sigmoid::new()));
        // 4th layer = 1 neuron - 4 inputs
        test.add_layer(NeuralLayer::new(1, 8, Sigmoid::new()));

        test.train(dataset, 1, 0.1f64);

        let think = test.evaluate(Sample::predict(vec![1f64, 0f64, 1f64]));

        assert_eq!(think.rows(), 1);
        assert_eq!(think.cols(), 1);
    }

    #[test]
    fn train_test_multiclass() {
        let dataset = vec![
            Sample::new(vec![1f64, 0f64, 2f64], vec![0f64, 1f64]),
            Sample::new(vec![1f64, 1f64, 5f64], vec![1f64, 0f64]),
        ];

        let mut test = NeuralNetwork::new();

        let sig_activation = Sigmoid::new();

        // 1st layer = 3 neurons - 2 inputs
        test.add_layer(NeuralLayer::new(3, 3, sig_activation));
        // 2nd layer = 1 neuron - 3 inputs
        test.add_layer(NeuralLayer::new(2, 3, sig_activation));

        test.train(dataset, 5, 0.1f64);
    }
}
