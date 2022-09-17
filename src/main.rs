use nalgebra::DMatrix;
use numerical::convolve;

fn main() {
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

    println!("{result}");
}
