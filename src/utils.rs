use sample::Sample;
use matrix::Matrix;
use matrix::MatrixTrait;

pub fn samples_input_to_matrix(samples: &Vec<Sample>) -> Matrix {
    let mut f64_vec: Vec<Vec<f64>> = vec![];

    for sample in samples.iter() {
        f64_vec.push(sample.inputs.clone());
    }

    return Matrix::generate(samples.len(), samples[0].get_inputs_count(), &|m,n| f64_vec[m][n]);
}

pub fn samples_output_to_matrix(samples: &Vec<Sample>) -> Matrix {
    let mut f64_vec: Vec<Vec<f64>> = vec![];

    for sample in samples.iter() {
        f64_vec.push(sample.outputs.clone());
    }

    return Matrix::generate(samples.len(), samples[0].get_outputs_count(), &|m,n| f64_vec[m][n]);
}
