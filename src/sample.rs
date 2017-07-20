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


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inputs_count() {
        let sample = Sample::new(vec![1f64, 0f64], vec![0f64]);
        assert_eq!(sample.get_inputs_count(), 2);
    }

    #[test]
    fn outputs_count() {
        let sample = Sample::new(vec![1f64, 0f64], vec![0f64]);
        assert_eq!(sample.get_outputs_count(), 1);
    }

    #[test]
    fn new_predict_inputs_count() {
        let sample = Sample::predict(vec![1f64, 0f64]);
        assert_eq!(sample.get_inputs_count(), 2);
    }

    #[test]
    fn new_predict_output_count() {
        let sample = Sample::predict(vec![1f64, 0f64]);
        assert_eq!(sample.get_outputs_count(), 0);
    }
}
