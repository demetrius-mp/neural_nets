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

pub fn predict(theta: &Matrix, x: &Matrix) -> f64 {
    theta.component_mul(x).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::create_matrix;

    struct Prediction {
        input: Matrix,
        output: f64
    }

    struct LogicalAndProblemData {
        x: Matrix,
        y: Matrix,
        initial_theta: Matrix,
        predictions: Vec<Prediction>
    }

    impl LogicalAndProblemData {
        fn new() -> Self {
            #[rustfmt::skip]
            let x = create_matrix(
                3,
                2,
                vec![
                1.0, 50.0,
                1.0, 60.0,
                1.0, 100.0,
            ]);

            #[rustfmt::skip]
            let y = create_matrix(
                3, 
                1, 
                vec![
                    120.0, 
                    150.0, 
                    250.0,
                ]
            );

            #[rustfmt::skip]
            let initial_theta = create_matrix(
                1,
                2,
                vec![
                    1.0, 1.0
                ]
            );

            let p1 = create_matrix(1, 2, vec![1.0, 50.0]);
            let p2 = create_matrix(1, 2, vec![1.0, 60.0]);
            let p3 = create_matrix(1, 2, vec![1.0, 100.0]);

            Self { 
                x, 
                y, 
                initial_theta, 
                predictions: vec![
                    Prediction {
                        input: p1,
                        output: 120.0,
                    },
                    Prediction {
                        input: p2,
                        output: 150.0,
                    },
                    Prediction {
                        input: p3,
                        output: 250.0,
                    },
                ] 
            }
        }
    }

    #[test]
    fn mini_batch_linear_regression_3_samples() {
        let data = LogicalAndProblemData::new();
        let theta = mini_batch_linear_regression(&data.x, &data.y, &data.initial_theta, 0.0001, 1000, 1);
        let mean_squared_error = data.predictions.iter().fold(0.0, |acc, p| {
            acc + (p.output - predict(&theta, &p.input)).powi(2)
        });

        assert!(mean_squared_error < 30.0);
    }
}
