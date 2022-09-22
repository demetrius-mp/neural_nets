use numerical::{
    create_matrix,
    stochastic_logistic_regression,
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

    let res = stochastic_logistic_regression(&x, &y, &initial_theta, 0.001, 10000);
    println!("{}", res);
}
