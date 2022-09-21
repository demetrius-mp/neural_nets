use nalgebra::DMatrix;
use numerical::{
    batch_linear_regression, 
    get_error, 
    mini_batch_linear_regression, 
    stochastic_linear_regression,
};

fn main() {
    #[rustfmt::skip]
    let x = DMatrix::from_row_iterator(
        3, 
        2, 
        vec![
        1.0, 50.0,
        1.0, 60.0,
        1.0, 100.0
    ]);

    #[rustfmt::skip]
    let y = DMatrix::from_row_iterator(
        3, 
        1, 
        vec![120.0, 150.0, 250.0]
    );

    let initial_theta = DMatrix::from_row_iterator(1, 2, vec![1.0, 1.0]);
    println!("{}", get_error(&x, &y, &initial_theta));

    let final_mini_batch_theta =
        mini_batch_linear_regression(&x, &y, &initial_theta, 0.0001, 1000, 1);
    let error = get_error(&x, &y, &final_mini_batch_theta);
    println!("{error}");

    let final_batch_theta = batch_linear_regression(&x, &y, &initial_theta, 0.0001, 1000);
    let error = get_error(&x, &y, &final_batch_theta);
    println!("{error}");

    let final_stochastic_theta = stochastic_linear_regression(&x, &y, &initial_theta, 0.0001, 1000);
    let error = get_error(&x, &y, &final_stochastic_theta);
    println!("{error}");
}
