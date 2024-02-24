use crate::layer::Dense;
use ndarray::Array2;

#[derive(Debug, Clone)]
pub enum Optimizer {
    Adam {
        lr: f64,
        // beta1: f64,
        // beta2: f64,
        // epsilon: f64,
        // m: Vec<Array2<f64>>,
        // v: Vec<Array2<f64>>,
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
        weight_updates: &[Array2<f64>],
        bias_updates: &[Array2<f64>],
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
    fn normalize_array(&self, a: &Array2<f64>) -> Option<Array2<f64>> {
        let avg = a.mean()?;
        Some(a / avg)
    }
}
