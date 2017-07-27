#[macro_use]
extern crate serde_derive;
extern crate csv;
extern crate juggernaut;

use std::fs::File;
use std::path::PathBuf;
use juggernaut::nl::NeuralLayer;
use juggernaut::nn::NeuralNetwork;
use juggernaut::activation::Activation;
use juggernaut::activation::Sigmoid;
use juggernaut::sample::Sample;
use juggernaut::matrix::MatrixTrait;

#[derive(Debug,Deserialize)]
enum FlowerClass {
    setosa,
    versicolor,
    virginica
}

#[derive(Debug,Deserialize)]
struct Flower {
    sepal_length: f64,
    sepal_width: f64,
    petal_length: f64,
    petal_width: f64,
    class: FlowerClass,
}

fn get_flower_class(class: FlowerClass) -> Vec<f64> {
    match class {
        FlowerClass::setosa => vec![0f64, 0f64, 1f64],
        FlowerClass::versicolor => vec![0f64, 1f64, 0f64],
        FlowerClass::virginica => vec![1f64, 0f64, 0f64]
    }
}

fn csv_to_dataset() -> Vec<Sample> {
    let mut dataset = vec![];

    let dataset_dir = PathBuf::from("./examples/dataset/iris.csv");

    let file = File::open(std::fs::canonicalize(&dataset_dir).expect("File not found")).unwrap();
    let mut rdr = csv::Reader::from_reader(file);

    for result in rdr.deserialize() {
        let flower: Flower = result.expect("Unable to convert the result");

        dataset.push(Sample::new(vec![flower.sepal_length, flower.sepal_width, flower.petal_length, flower.petal_width], get_flower_class(flower.class)))
    }

    println!("{:?}", dataset);

    dataset
}

fn main() {
    println!("Juggernaut...");

    let dataset = csv_to_dataset();

    println!("Creating the network...");

    let mut test = NeuralNetwork::new(dataset, Sigmoid::new());

    // 1st layer = 2 neurons - 3 inputs
    test.add_layer(NeuralLayer::new(7, 4));

    println!("First layer created: 2 neurons 3 inputs");

    // 2nd layer = 1 neuron - 2 inputs
    test.add_layer(NeuralLayer::new(3, 7));

    println!("Second layer created: 1 neuron 2 inputs");

    println!("Training (60,000 epochs)...");

    test.train(10000, 0.01f64);

    let think = test.evaluate(Sample::predict(vec![5f64,3.4f64,1.5f64,0.2f64]));

    println!("Evaluate [0, 0, 1] = {:?}", think);

    let think2 = test.evaluate(Sample::predict(vec![7.0f64,3.2f64,4.7f64,1.4f64]));

    println!("Evaluate [0, 1, 0] = {:?}", think2);

    let think3 = test.evaluate(Sample::predict(vec![6.2f64,3.4f64,5.4f64,2.3f64]));

    println!("Evaluate [1, 0, 0] = {:?}", think3);
}
