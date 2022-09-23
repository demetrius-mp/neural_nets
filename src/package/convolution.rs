use crate::{create_matrix, Matrix};

/// Runs a convolution on the given channels, using the given kernel.
/// # Arguments
///
/// * `kernel` - A vector of matrices, where each matrix will be convoluted with a channel of the image.
/// * `channels` - A vector of matrices, where each matrix represents a channel of the image.
///
/// # Examples
///
/// ```
/// use neural_nets::{convolution, create_matrix};
///
/// #[rustfmt::skip]
/// let kernel = create_matrix(3, 3, vec![
///     0.0, 1.0, 2.0,
///     3.0, 4.0, 5.0,
///     6.0, 7.0, 8.0,
/// ]);
///
/// #[rustfmt::skip]
/// let channel = create_matrix(4, 4, vec![
///     0.0, 1.0, 0.0, 0.0,
///     1.0, 1.0, 1.0, 1.0,
///     0.0, 1.0, 1.0, 1.0,
///     1.0, 0.0, 0.0, 1.0,
/// ]);
///
/// let result = convolution(
///     &vec![kernel],
///     &vec![channel],
/// );
///
/// #[rustfmt::skip]
/// let expected = create_matrix(2, 2, vec![
///     28.0, 33.0,
///     18.0, 23.0
/// ]);
///
/// assert_eq!(result, expected);
/// ```
pub fn convolution(kernel: &Vec<Matrix>, channels: &Vec<Matrix>) -> Matrix {
    let nrows = channels[0].nrows() - kernel[0].nrows() + 1;
    let ncols = channels[0].ncols() - kernel[0].ncols() + 1;

    let mut convoluted = create_matrix(nrows, ncols, vec![0.0; nrows * ncols]);

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::create_matrix;

    #[test]
    fn convolution_single_channel() {
        #[rustfmt::skip]
        let kernel = create_matrix(3, 3, vec![
            0.0, 1.0, 2.0,
            3.0, 4.0, 5.0,
            6.0, 7.0, 8.0,
        ]);

        #[rustfmt::skip]
        let channel = create_matrix(
            4,
            4,
            vec![
                0.0, 1.0, 0.0, 0.0,
                1.0, 1.0, 1.0, 1.0,
                0.0, 1.0, 1.0, 1.0,
                1.0, 0.0, 0.0, 1.0,
            ],
        );

        let result = convolution(&vec![kernel], &vec![channel]);

        #[rustfmt::skip]
        let expected = create_matrix(2, 2, vec![
            28.0, 33.0,
            18.0, 23.0
        ]);

        assert_eq!(result, expected);
    }

    #[test]
    fn convolution_multi_channel() {
        #[rustfmt::skip]
        let kernel = create_matrix(3, 3, vec![
            0.0, 1.0, 2.0,
            3.0, 4.0, 5.0,
            6.0, 7.0, 8.0,
        ]);

        #[rustfmt::skip]
        let kernel2 = create_matrix(3, 3, vec![
            9.0, 10.0, 11.0,
            12.0, 13.0, 14.0,
            15.0, 16.0, 17.0
        ]);

        #[rustfmt::skip]
        let kernel3 = create_matrix(3, 3, vec![
            18.0, 19.0, 20.0,
            21.0, 22.0, 23.0,
            24.0, 25.0, 26.0
        ]);

        #[rustfmt::skip]
        let channel = create_matrix(
            4,
            4,
            vec![
                0.0, 1.0, 0.0, 0.0,
                1.0, 1.0, 1.0, 1.0,
                0.0, 1.0, 1.0, 1.0,
                1.0, 0.0, 0.0, 1.0,
            ],
        );

        #[rustfmt::skip]
        let channel2 = create_matrix(
            4,
            4,
            vec![
                0.0, 0.0, 0.0, 1.0,
                0.0, 1.0, 0.0, 1.0,
                1.0, 0.0, 0.0, 0.0,
                1.0, 1.0, 1.0, 1.0,
            ],
        );

        #[rustfmt::skip]
        let channel3 = create_matrix(
            4,
            4,
            vec![
                1.0, 1.0, 0.0, 0.0,
                0.0, 1.0, 1.0, 0.0,
                0.0, 1.0, 1.0, 1.0,
                1.0, 0.0, 1.0, 0.0,
            ],
        );

        let result = convolution(
            &vec![kernel, kernel2, kernel3],
            &vec![channel, channel2, channel3],
        );

        #[rustfmt::skip]
        let expected = create_matrix(2, 2, vec![
            189.0, 206.0,
            222.0, 219.0
        ]);

        assert_eq!(result, expected);
    }
}
