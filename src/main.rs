use neural_nets::create_matrix;

use neural_nets::linear_regression::{
    LinearRegression
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

    let lr = LinearRegression::new(&x, &y, &initial_theta, 0.0001, 1000);
    let res = lr.fit(1);
    println!("{}", res);

    println!(
        "{}",
        LinearRegression::predict(&res, &create_matrix(1, 2, vec![1.0, 60.0]))
    );
}
