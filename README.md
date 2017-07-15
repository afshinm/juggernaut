# Juggernaut [![Build Status](https://travis-ci.org/afshinm/juggernaut.svg?branch=master)](https://travis-ci.org/afshinm/juggernaut)
> Juggernaut is an experimental Neural Network written in Rust

<img src="http://juggernaut.rs/static/images/art.png" alt="hi" class="inline"/>

# Example

Want to setup a simple network using Juggernaut? 

This sample creates a random binary operation network with one hidden layer:

```
fn main() {
    let dataset = vec![
        Sample::new(vec![0f64, 0f64, 1f64], vec![0f64]),
        Sample::new(vec![0f64, 1f64, 1f64], vec![0f64]),
        Sample::new(vec![1f64, 0f64, 1f64], vec![1f64]),
        Sample::new(vec![1f64, 1f64, 1f64], vec![1f64])
    ];

    let think_dataset = vec![
        Sample::new(vec![1f64, 0f64, 1f64], vec![0f64])
    ];

    let mut test = NeuralNetwork::new(dataset, Sigmoid::new());

    // 1st layer = 2 neurons - 3 inputs
    test.add_layer(NeuralLayer::new(2, 3));
    // 2nd layer = 1 neuron - 2 inputs
    test.add_layer(NeuralLayer::new(1, 2));

    test.train(60000);

    let think = test.forward(&think_dataset);
}

```

and the output of `think` is the prediction of the network after training with 60,000 epochs.

# Test

Install Rust 1.x and run:

```
cargo test
```

# Author

Afshin Mehrabani (afshin.meh@gmail.com)

# FAQ

### Contributing

Fork the project and send PRs + unit tests for that specific part. 

### "Juggernaut"?

Juggernaut is a Dota2 hero and I like this hero. Juggernaut is a powerful hero, when he has enough farm.

# License

GNU General Public License v3.0
