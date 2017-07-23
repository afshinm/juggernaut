# Juggernaut [![Build Status](https://travis-ci.org/afshinm/juggernaut.svg?branch=master)](https://travis-ci.org/afshinm/juggernaut)
> Juggernaut is an experimental Neural Network written in Rust

<img src="http://juggernaut.rs/static/images/art.png" alt="hi" class="inline"/>

# Example

Want to setup a simple network using Juggernaut? 

This sample creates a random binary operation network with one hidden layer:

```rust
fn main() {
    let dataset = vec![
        Sample::new(vec![0f64, 0f64, 1f64], vec![0f64]),
        Sample::new(vec![0f64, 1f64, 1f64], vec![0f64]),
        Sample::new(vec![1f64, 0f64, 1f64], vec![1f64]),
        Sample::new(vec![1f64, 1f64, 1f64], vec![1f64])
    ];
    
    let mut test = NeuralNetwork::new(dataset, Sigmoid::new());

    // 1st layer = 2 neurons - 3 inputs
    test.add_layer(NeuralLayer::new(2, 3));

    // 2nd layer = 1 neuron - 2 inputs
    test.add_layer(NeuralLayer::new(1, 2));

    test.error(|err| {
        println!("error({})", err.to_string());
    });

    test.train(10000);
    
    let think = test.evaluate(Sample::predict(vec![1f64, 0f64, 1f64]));

    println!("Evaluate [1, 0, 1] = {:?}", think.get(0, 0));

}

```

and the output of `think` is the prediction of the network after training.

# Documentation

https://docs.rs/juggernaut

# Build

To build the demo, run:

```
cargo build --example helloworld --verbose
```

then to run the compiled file:

```
./target/debug/examples/helloworld
```

# Test

Install Rust 1.x and run:

```
cargo test
```

# Author

Afshin Mehrabani (afshin.meh@gmail.com) and [contributors](https://github.com/afshinm/juggernaut/graphs/contributors)

# FAQ

### Contributing

Fork the project and send PRs + unit tests for that specific part. 

### "Juggernaut"?

Juggernaut is a Dota2 hero and I like this hero. Juggernaut is a powerful hero, when he has enough farm.

# License

GNU General Public License v3.0

<script>
  (function(i,s,o,g,r,a,m){i['GoogleAnalyticsObject']=r;i[r]=i[r]||function(){
  (i[r].q=i[r].q||[]).push(arguments)},i[r].l=1*new Date();a=s.createElement(o),
  m=s.getElementsByTagName(o)[0];a.async=1;a.src=g;m.parentNode.insertBefore(a,m)
  })(window,document,'script','https://www.google-analytics.com/analytics.js','ga');

  ga('create', 'UA-103062405-1', 'auto');
  ga('send', 'pageview');

</script>
