use crate::Matrix;

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

pub struct LogisticRegression<'a> {
    x: &'a Matrix,
    y: &'a Matrix,
    initial_theta: &'a Matrix,
    alpha: f64,
    epochs: u128,
}

impl<'a> LogisticRegression<'a> {
    pub fn new(
        x: &'a Matrix,
        y: &'a Matrix,
        initial_theta: &'a Matrix,
        alpha: f64,
        epochs: u128,
    ) -> Self {
        Self { x, y, initial_theta, alpha, epochs }
    }

    pub fn fit(&self, mini_batch_size: usize) -> Matrix {
        let mut theta = self.initial_theta.clone();

        for _ in 0..self.epochs {
            for i in (0..self.x.nrows()).step_by(mini_batch_size) {
                let x_mini_batch = self.x.rows(i, mini_batch_size);
                let y_mini_batch = self.y.rows(i, mini_batch_size);

                let factor = 1.0 / mini_batch_size as f64;

                let theta_x_t = &theta * &x_mini_batch.transpose();
                let gx = &theta_x_t.map(sigmoid);
                let gx_minus_y = gx - &y_mini_batch.transpose();
                let one_minus_gx = &gx.map(|v| 1.0 - v);

                let mut delta = gx_minus_y.component_mul(gx).component_mul(one_minus_gx);
                delta = delta * x_mini_batch;
                delta = delta.scale(factor);

                theta -= self.alpha * delta;
            }
        }

        theta
    }

    pub fn predict(theta: &Matrix, x: &Matrix) -> bool {
        let res = sigmoid(theta.component_mul(x).sum());
    
        res > 0.5
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::create_matrix;

    struct Prediction {
        input: Matrix,
        output: bool
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
                4, 
                3, 
                vec![
                1.0, 0.0, 0.0,
                1.0, 0.0, 1.0,
                1.0, 1.0, 0.0,
                1.0, 1.0, 1.0,
            ]);

            #[rustfmt::skip]
            let y = create_matrix(
                4, 
                1, 
                vec![
                    0.0, 
                    0.0, 
                    0.0,
                    1.0,
                ]
            );

            #[rustfmt::skip]
            let initial_theta = create_matrix(
                1, 
                3, 
                vec![
                    1.0, 1.0, 1.0
                ]
            );

            let p1 = create_matrix(1, 3, vec![1.0, 0.0, 0.0]);
            let p2 = create_matrix(1, 3, vec![1.0, 0.0, 1.0]);
            let p3 = create_matrix(1, 3, vec![1.0, 1.0, 0.0]);
            let p4 = create_matrix(1, 3, vec![1.0, 1.0, 1.0]);

            Self { 
                x, 
                y, 
                initial_theta, 
                predictions: vec![
                    Prediction {
                        input: p1,
                        output: false,
                    },
                    Prediction {
                        input: p2,
                        output: false,
                    },
                    Prediction {
                        input: p3,
                        output: false,
                    },
                    Prediction {
                        input: p4,
                        output: true,
                    },
                ] 
            }
        }
    }

    #[test]
    fn mini_batch_logistic_regression_on_logical_and_problem() {
        let data = LogicalAndProblemData::new();
        let lr = LogisticRegression::new(&data.x, &data.y, &data.initial_theta, 0.5, 1000);
        let theta = lr.fit(1);

        for prediction in data.predictions {
            let result = LogisticRegression::predict(&theta, &prediction.input);
            assert_eq!(result, prediction.output);
        }
    }
}
