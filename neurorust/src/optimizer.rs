use crate::{gradient::Grad, loss::Loss, stage::Trainable};
use std::ops::{Mul, Sub};

pub trait Optimizer<Data, Target, Parameters>
where
    Parameters: Mul<f64, Output = Parameters> + Sub,
{
    fn update<T>(
        self,
        model: Grad<Parameters = Parameters, Output = Target>,
        loss: impl Loss,
        data: Data,
    ) -> Parameters;
}
#[derive(Debug, Clone)]
pub struct Adam {
    lr: f64,
    // beta1: f64,
    // beta2: f64,
    // epsilon: f64,
    // m: Vec<Matrix>,
    // v: Vec<Matrix>,
}
#[derive(Debug, Clone)]
pub struct SGD {
    lr: f64,
}
#[derive(Debug, Clone)]
pub struct GD {
    pub epochs: u64,
    pub lr: f64,
}

impl<Data, Target, Parameters> Optimizer<Data, Target, Parameters> for GD
where
    Parameters: Mul<f64, Output = Parameters> + Sub<Output = Parameters> + Copy,
{
    fn update<T>(
        self,
        model: Grad<Parameters = Parameters, Output = Target>,
        loss: impl Loss,
        data: Data,
    ) -> Parameters {
        let (da_dW, da_da) = model.grad(data);
        let dL_da = 5.0;
        *model.parameters() - dL_da * da_dW * self.lr
    }
}
// fn update<T>(self, model: Trainable, data: Data) -> Parameters {
// let output = model.forward(data);
// let loss = self.loss

// TODO - Consider normalizing the updates
// for (i, layer) in net.weights().iter_mut().enumerate() {
//     layer.weights.as_ref() -= &(*lr * &weight_updates[i]);
//     layer.biases.as_ref() -= &(*lr * &bias_updates[i]);
// }
// }

// #[allow(dead_code)]
// fn normalize_array(&self, a: &Matrix) -> Option<Matrix> {
//     let avg = a.mean()?;
//     Some(a / avg)
// }
// }
