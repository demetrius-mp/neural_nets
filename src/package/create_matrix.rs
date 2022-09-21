use nalgebra::{DMatrix, Dynamic, Matrix as NMatrix, VecStorage};

pub type Matrix = NMatrix<f64, Dynamic, Dynamic, VecStorage<f64, Dynamic, Dynamic>>;

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
            1.0, 1.0, 1.0
        ];

        let result = create_matrix(nrows, ncols, data.to_vec());

        let expected = DMatrix::from_row_iterator(nrows, ncols, data.to_vec());

        assert_eq!(result, expected);
    }

    #[test]
    fn create_matrix_single_column() {
        let nrows = 1;
        let ncols = 3;

        #[rustfmt::skip]
        let data = vec![
            1.0,
            1.0,
            1.0
        ];

        let result = create_matrix(nrows, ncols, data.to_vec());

        let expected = DMatrix::from_row_iterator(nrows, ncols, data.to_vec());

        assert_eq!(result, expected);
    }

    #[test]
    fn create_matrix_multiple_rows_and_columns() {
        let nrows = 3;
        let ncols = 3;

        #[rustfmt::skip]
        let data = vec![
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
        ];

        let result = create_matrix(nrows, ncols, data.to_vec());

        let expected = DMatrix::from_row_iterator(nrows, ncols, data.to_vec());

        assert_eq!(result, expected);
    }
}
