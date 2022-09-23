use crate::Matrix;

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

pub fn batch_logistic_regression(
    x: &Matrix,
    y: &Matrix,
    initial_theta: &Matrix,
    alpha: f64,
    epochs: u128,
) -> Matrix {
    let mut theta = initial_theta.clone();

    let x_transposed = x.transpose();
    let y_transposed = y.transpose();
    let factor = 1.0 / x.nrows() as f64;

    for _ in 0..epochs {
        let theta_x_t = &theta * &x_transposed;
        let gx = &theta_x_t.map(sigmoid);
        let gx_minus_y = gx - &y_transposed;
        let one_minus_gx = &gx.map(|v| 1.0 - v);

        let mut delta = gx_minus_y.component_mul(gx).component_mul(one_minus_gx);
        delta = delta * x;
        delta = delta.scale(factor);

        theta = theta - alpha * delta;
    }

    theta
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
                let delta = (gx - y[i]) * gx * (1.0 - gx) * x[(i, j)];
                theta[j] = theta[j] - (alpha * delta);
            }
        }
    }

    theta
}

pub fn predict(theta: &Matrix, x: &Matrix) -> bool {
    let res = sigmoid(theta.component_mul(x).sum());

    res > 0.5
}
