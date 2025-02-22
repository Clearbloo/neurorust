use crate::layer::Dense;
use crate::utils::Matrix;

#[derive(Debug, Clone)]
pub enum Optimizer {
    Adam {
        lr: f64,
        // beta1: f64,
        // beta2: f64,
        // epsilon: f64,
        // m: Vec<Matrix>,
        // v: Vec<Matrix>,
    },
    SGD {
        lr: f64,
    },
    GD {
        lr: f64,
    },
}

impl Optimizer {
    pub fn apply_updates(
        &self,
        layers: &mut [Dense],
        weight_updates: &[Matrix],
        bias_updates: &[Matrix],
    ) {
        match self {
            Self::Adam { .. } => todo!(),
            Self::SGD { .. } => todo!(),
            Self::GD { lr } => {
                // TODO - Consider normalizing the updates
                for (i, layer) in layers.iter_mut().enumerate() {
                    layer.weights.data -= &(*lr * &weight_updates[i]);
                    layer.biases.data -= &(*lr * &bias_updates[i]);
                }
            }
        }
    }

    #[allow(dead_code)]
    fn normalize_array(&self, a: &Matrix) -> Option<Matrix> {
        let avg = a.mean()?;
        Some(a / avg)
    }
}
