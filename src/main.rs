use neural_nets::create_matrix;

use neural_nets::linear_regression::{
    mini_batch_linear_regression,
    predict,
};

fn main() {
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

    let res = mini_batch_linear_regression(&x, &y, &initial_theta, 0.0001, 1000, 1);
    println!("{}", res);

    println!(
        "{}",
        predict(&res, &create_matrix(1, 2, vec![1.0, 60.0]))
    );
}
