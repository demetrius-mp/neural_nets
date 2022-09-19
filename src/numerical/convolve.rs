use nalgebra::{DMatrix, Dynamic, Matrix, VecStorage};

/// Runs a convolution on the given channels, using the given kernel.
/// # Arguments
///
/// * `kernel` - A vector of matrices, where each matrix will be convolved with a channel of the image.
/// * `channels` - A vector of matrices, where each matrix represents a channel of the image.
///
/// # Examples
///
/// ```
/// use numerical::convolve;
/// use nalgebra::{DMatrix};
///
/// #[rustfmt::skip]
/// let kernel = DMatrix::from_vec(3, 3, vec![
///     0.0, 1.0, 2.0,
///     3.0, 4.0, 5.0,
///     6.0, 7.0, 8.0,
/// ]).transpose();
///
/// #[rustfmt::skip]
/// let channel = DMatrix::from_vec(
///     4,
///     4,
///     vec![
///         0.0, 1.0, 0.0, 0.0,
///         1.0, 1.0, 1.0, 1.0,
///         0.0, 1.0, 1.0, 1.0,
///         1.0, 0.0, 0.0, 1.0,
///     ],
/// ).transpose();
///
/// let result = convolve(
///     &vec![kernel],
///     &vec![channel],
/// );
///
/// #[rustfmt::skip]
/// let expected = DMatrix::from_vec(2, 2, vec![
///     28.0, 33.0,
///     18.0, 23.0
/// ]).transpose();
///
/// assert_eq!(result, expected);
/// ```
pub fn convolve(kernel: &Vec<Container>, channels: &Vec<Container>) -> Container {
    let nrows = channels[0].nrows() - kernel[0].nrows() + 1;
    let ncols = channels[0].ncols() - kernel[0].ncols() + 1;

    let mut convoluted = DMatrix::from_vec(nrows, ncols, vec![0.0; nrows * ncols]);

    for i in 0..nrows {
        for j in 0..ncols {
            for (kernel_channel, channel) in kernel.iter().zip(channels) {
                let channel_slice =
                    channel.slice((i, j), (kernel_channel.nrows(), kernel_channel.ncols()));
                let product = kernel_channel.component_mul(&channel_slice);

                convoluted[(i, j)] += product.sum();
            }
        }
    }

    convoluted
}

type Container = Matrix<f64, Dynamic, Dynamic, VecStorage<f64, Dynamic, Dynamic>>;

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;

    #[test]
    fn convolve_single_channel() {
        #[rustfmt::skip]
        let kernel = DMatrix::from_vec(3, 3, vec![
            0.0, 1.0, 2.0,
            3.0, 4.0, 5.0,
            6.0, 7.0, 8.0,
        ]).transpose();

        #[rustfmt::skip]
        let channel = DMatrix::from_vec(
            4,
            4,
            vec![
                0.0, 1.0, 0.0, 0.0,
                1.0, 1.0, 1.0, 1.0,
                0.0, 1.0, 1.0, 1.0,
                1.0, 0.0, 0.0, 1.0,
            ],
        ).transpose();

        let result = convolve(&vec![kernel], &vec![channel]);

        #[rustfmt::skip]
        let expected = DMatrix::from_vec(2, 2, vec![
            28.0, 33.0,
            18.0, 23.0
        ]).transpose();

        assert_eq!(result, expected);
    }

    #[test]
    fn convolve_multi_channel() {
        #[rustfmt::skip]
        let kernel = DMatrix::from_vec(3, 3, vec![
            0.0, 1.0, 2.0,
            3.0, 4.0, 5.0,
            6.0, 7.0, 8.0,
        ]).transpose();

        #[rustfmt::skip]
        let kernel2 = DMatrix::from_vec(3, 3, vec![
            9.0, 10.0, 11.0,
            12.0, 13.0, 14.0,
            15.0, 16.0, 17.0
        ]).transpose();

        #[rustfmt::skip]
        let kernel3 = DMatrix::from_vec(3, 3, vec![
            18.0, 19.0, 20.0,
            21.0, 22.0, 23.0,
            24.0, 25.0, 26.0
        ]).transpose();

        #[rustfmt::skip]
        let channel = DMatrix::from_vec(
            4,
            4,
            vec![
                0.0, 1.0, 0.0, 0.0,
                1.0, 1.0, 1.0, 1.0,
                0.0, 1.0, 1.0, 1.0,
                1.0, 0.0, 0.0, 1.0,
            ],
        ).transpose();

        #[rustfmt::skip]
        let channel2 = DMatrix::from_vec(
            4,
            4,
            vec![
                0.0, 0.0, 0.0, 1.0,
                0.0, 1.0, 0.0, 1.0,
                1.0, 0.0, 0.0, 0.0,
                1.0, 1.0, 1.0, 1.0,
            ],
        ).transpose();

        #[rustfmt::skip]
        let channel3 = DMatrix::from_vec(
            4,
            4,
            vec![
                1.0, 1.0, 0.0, 0.0,
                0.0, 1.0, 1.0, 0.0,
                0.0, 1.0, 1.0, 1.0,
                1.0, 0.0, 1.0, 0.0,
            ],
        ).transpose();

        let result = convolve(
            &vec![kernel, kernel2, kernel3],
            &vec![channel, channel2, channel3],
        );

        #[rustfmt::skip]
        let expected = DMatrix::from_vec(2, 2, vec![
            189.0, 206.0,
            222.0, 219.0
        ]).transpose();

        assert_eq!(result, expected);
    }
}
