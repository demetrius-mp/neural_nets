use crate::Matrix;

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

pub fn stochastic_logistic_regression(
    x: &Matrix,
    y: &Matrix,
    initial_theta: &Matrix,
    alpha: f64,
    epochs: u128,
) -> Matrix {
    let mut theta = initial_theta.clone();

    for _ in 0..epochs {
        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                let gx = sigmoid(theta.component_mul(&x.row(i)).sum());
                let delta = (gx - y[i]) * (gx * (1.0 - gx)) * x[(i, j)];
                theta[j] = theta[j] - (alpha * delta);
            }
        }
    }

    theta
}
