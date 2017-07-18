/// A sample of the given dataset

#[derive(Debug)]
pub struct Sample {
    pub inputs: Vec<f64>,
    pub outputs: Option<Vec<f64>>
}

impl Sample {
    pub fn new(inputs: Vec<f64>, outputs: Vec<f64>) -> Sample {
        Sample{
            inputs: inputs,
            outputs: Some(outputs)
        }
    }

    pub fn predict(inputs: Vec<f64>) -> Sample {
        Sample{
            inputs: inputs,
            outputs: None
        }
    }

    pub fn get_inputs_count(&self) -> usize {
        self.inputs.len()
    }

    pub fn get_outputs_count(&self) -> usize {
        match &self.outputs {
            &Some(ref outputs) => outputs.len(),
            &None => 0
        }
    }
}
