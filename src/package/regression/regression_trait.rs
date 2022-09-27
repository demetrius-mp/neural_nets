use crate::Matrix;

pub trait RegressionTrait<'a> {
    fn new(
        x: &'a Matrix,
        y: &'a Matrix,
        initial_theta: &'a Matrix,
        alpha: f64,
        epochs: u128,
    ) -> Self;
    fn fit(&self, mini_batch_size: usize) -> Matrix;
    fn predict(theta: &Matrix, x: &Matrix) -> f64;
}
