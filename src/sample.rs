/// A sample of the given dataset

pub struct Sample {
    pub inputs: Vec<f64>,
    pub outputs: Vec<f64>
}

impl Sample {
    pub fn new(inputs: Vec<f64>, outputs: Vec<f64>) -> Sample {
        Sample{
            inputs: inputs,
            outputs: outputs
        }
    }

    pub fn get_inputs_count(&self) -> usize {
        self.inputs.len()
    }

    pub fn get_outputs_count(&self) -> usize {
        self.outputs.len()
    }
}
