use nalgebra::{DMatrix, Dynamic, Matrix as NMatrix, VecStorage};

pub type Matrix = NMatrix<f64, Dynamic, Dynamic, VecStorage<f64, Dynamic, Dynamic>>;

/// Creates a matrix. It is used as entry for many functions of this package.
///
/// This is just a wrapper for running  `DMatrix::from_row_iterator(nrows, ncols, data)`
///
/// # Arguments
///
/// * `nrows` - Number of rows of the matrix.
/// * `ncols` - Number of columns of the matrix.
/// * `data` - Vec<f64> containing the entries of the matrix.
///
/// # Examples
///
/// ```
/// use numerical::create_matrix;
///
/// let nrows = 3;
/// let ncols = 3;
///
/// #[rustfmt::skip]
/// let data = vec![
///     1.0, 2.0, 3.0,
///     4.0, 5.0, 6.0,
///     7.0, 8.0, 9.0,
/// ];
///
/// let m = create_matrix(nrows, ncols, data);
///
/// assert_eq!(m[(1, 1)], 5.0);
/// ```
pub fn create_matrix(nrows: usize, ncols: usize, data: Vec<f64>) -> Matrix {
    DMatrix::from_row_iterator(nrows, ncols, data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;

    #[test]
    fn create_matrix_single_row() {
        let nrows = 1;
        let ncols = 3;

        #[rustfmt::skip]
        let data = vec![
            1.0, 2.0, 3.0
        ];

        let result = create_matrix(nrows, ncols, data.to_vec());

        let expected = DMatrix::from_row_iterator(nrows, ncols, data.to_vec());

        assert_eq!(result, expected);
        assert_eq!(result[1], 2.0);
    }

    #[test]
    fn create_matrix_single_column() {
        let nrows = 1;
        let ncols = 3;

        #[rustfmt::skip]
        let data = vec![
            1.0,
            2.0,
            3.0
        ];

        let result = create_matrix(nrows, ncols, data.to_vec());

        let expected = DMatrix::from_row_iterator(nrows, ncols, data.to_vec());

        assert_eq!(result, expected);
        assert_eq!(result[1], 2.0);
    }

    #[test]
    fn create_matrix_multiple_rows_and_columns() {
        let nrows = 3;
        let ncols = 3;

        #[rustfmt::skip]
        let data = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ];

        let result = create_matrix(nrows, ncols, data.to_vec());

        let expected = DMatrix::from_row_iterator(nrows, ncols, data.to_vec());

        assert_eq!(result, expected);
        assert_eq!(result[(1, 1)], 5.0);
    }
}
