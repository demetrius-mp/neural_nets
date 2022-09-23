use crate::Matrix;

pub fn mini_batch_linear_regression(
    x: &Matrix,
    y: &Matrix,
    initial_theta: &Matrix,
    alpha: f64,
    epochs: u128,
    mini_batch_size: usize,
) -> Matrix {
    let mut theta = initial_theta.clone();

    for _ in 0..epochs {
        for i in (0..x.nrows()).step_by(mini_batch_size) {
            let x_mini_batch = x.rows(i, mini_batch_size);
            let y_mini_batch = y.rows(i, mini_batch_size);
            let x_mini_batch_mean_values = x_mini_batch.scale(1.0 / mini_batch_size as f64);

            let current_guess_distance = &theta * &x_mini_batch.transpose() - &y_mini_batch;

            let delta = current_guess_distance * &x_mini_batch_mean_values;

            theta -= alpha * delta;
        }
    }

    theta
}

pub fn batch_linear_regression(
    x: &Matrix,
    y: &Matrix,
    initial_theta: &Matrix,
    alpha: f64,
    epochs: u128,
) -> Matrix {
    let mut theta = initial_theta.clone();

    let number_of_samples = x.nrows();

    let x_transposed = x.transpose();
    let y_transposed = y.transpose();
    let x_mean_values = x.scale(1.0 / number_of_samples as f64);

    for _ in 0..epochs {
        let delta = (&theta * &x_transposed - &y_transposed) * &x_mean_values;
        theta = theta - alpha * delta;
    }

    theta
}

pub fn stochastic_linear_regression(
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
                let delta = (theta.component_mul(&x.row(i)).sum() - y[i]) * x[(i, j)];
                theta[j] = theta[j] - (alpha * delta);
            }
        }
    }

    theta
}

pub fn predict(theta: &Matrix, x: &Matrix) -> f64 {
    theta.component_mul(x).sum()
}
