use std::ops::Mul;

use crate::stage::{Predictor, Trainable};

// TODO - Rename to differentiable
pub trait Grad: Predictor {
    fn grad(&self, data: Self::Input) -> (Self::Parameters, Self::Output);
}
