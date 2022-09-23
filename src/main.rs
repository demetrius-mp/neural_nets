use numerical::create_matrix;

use numerical::logistic_regression::{
    batch_logistic_regression, 
    mini_batch_logistic_regression, 
    stochastic_logistic_regression,
    predict,
};

fn main() {
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

    let res = batch_logistic_regression(&x, &y, &initial_theta, 0.5, 1000);
    println!("{}", res);

    let res = stochastic_logistic_regression(&x, &y, &initial_theta, 0.5, 1000);
    println!("{}", res);

    let res = mini_batch_logistic_regression(&x, &y, &initial_theta, 0.5, 1000, 1);
    println!("{}", res);

    println!(
        "{}",
        predict(&res, &create_matrix(1, 3, vec![1.0, 1.0, 1.0]))
    );
}
