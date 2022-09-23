type Derivative = fn(&Vec<f64>) -> f64;

pub enum GradientMode {
    Asc,
    Desc,
}

/// Runs the gradient method using the given parameters.
/// # Arguments
///
/// * `initial_values` - The initial values to start the method.
/// * `derivatives` - A vector of derivatives, where derivative\[i\] is the derivative of the function
/// with respect to initial_values\[i\].
/// * `alpha` - The size of the "step" for each iteration.
/// * `epochs` - Number of iterations.
/// * `mode` - Wheter to run the gradient in ascent, or descent mode.
///
/// # Examples
///
/// ```
/// use neural_nets::{gradient, GradientMode};
///
/// // runs the gradient descent method on the function
/// // f(x, y) = x**2 + y**2
/// let result = gradient(
///     &vec![5.0, 8.0],
///     &vec![|values| 2.0 * values[0], |values| 2.0 * values[1]],
///     0.1,
///     1200,
///     GradientMode::Desc,
/// );
///
/// let expected = vec![0.0, 0.0];
/// let difference_x = (expected[0] - result[0]).abs();
/// let difference_y = (expected[1] - result[1]).abs();
///
/// assert!(difference_x < 1e-100);
/// assert!(difference_y < 1e-100);
/// ```
pub fn gradient(
    initial_values: &Vec<f64>,
    derivatives: &Vec<Derivative>,
    alpha: f64,
    epochs: u128,
    mode: GradientMode,
) -> Vec<f64> {
    let mut values = initial_values.to_vec();
    for _ in 0..epochs {
        for i in 0..initial_values.len() {
            let delta = derivatives[i](&values);
            match mode {
                GradientMode::Asc => values[i] += alpha * delta,
                GradientMode::Desc => values[i] -= alpha * delta,
            }
        }
    }

    values
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gradient_single_variable_asc() {
        let result = gradient(
            &vec![0.0],
            &vec![|values| {
                let sigmoid = |x: f64| 1.0 / (1.0 + x.exp());
                let result = sigmoid(values[0]);

                result * (1.0 - result)
            }],
            0.5,
            3,
            GradientMode::Asc,
        );

        let expected = 0.3725874750366366;
        let difference = (expected - result[0]).abs();

        assert!(difference < 1e-10);
    }

    #[test]
    fn gradient_single_variable_desc() {
        let result = gradient(
            &vec![0.0],
            &vec![|values| {
                let sigmoid = |x: f64| 1.0 / (1.0 + x.exp());
                let result = sigmoid(values[0]);

                result * (1.0 - result)
            }],
            0.5,
            3,
            GradientMode::Desc,
        );

        let expected = -0.3725874750366366;
        let difference = (expected - result[0]).abs();

        assert!(difference < 1e-10);
    }

    #[test]
    fn gradient_multi_variable_desc() {
        let result = gradient(
            &vec![5.0, 8.0],
            &vec![|values| 2.0 * values[0], |values| 2.0 * values[1]],
            0.1,
            1200,
            GradientMode::Desc,
        );

        let expected = vec![0.0, 0.0];
        let difference_x = (expected[0] - result[0]).abs();
        let difference_y = (expected[1] - result[1]).abs();

        assert!(difference_x < 1e-100);
        assert!(difference_y < 1e-100);
    }

    #[test]
    fn gradient_multi_variable_asc() {
        let result = gradient(
            &vec![3.5, 3.5],
            &vec![|values| -2.0 * values[0], |values| -2.0 * values[1]],
            0.1,
            1200,
            GradientMode::Asc,
        );

        let expected = vec![0.0, 0.0];
        let difference_x = (expected[0] - result[0]).abs();
        let difference_y = (expected[1] - result[1]).abs();

        assert!(difference_x < 1e-100);
        assert!(difference_y < 1e-100);
    }
}
