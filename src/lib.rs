/// Equivalent to running the dot product between `a` and `b`,
/// then summing every entry of the result.
/// - If one of the entries is empty, it returns 0.0.
/// - If the length of the entries is different, it goes up to the smallest one.
///
/// # Arguments
///
/// * `a` - A `Vector<f64>` containing 0 or more items.
/// * `b` - A `Vector<f64>` containing 0 or more items.
///
/// # Examples
///
/// ```
/// use numerical::dot_product_and_sum;
///
/// let a: Vec<f64> = vec![1.0, 2.0];
/// let b: Vec<f64> = vec![3.0, 4.0];
/// let result = dot_product_and_sum(&a, &b);
/// assert_eq!(result, 11.0);
/// ```
pub fn dot_product_and_sum(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
    a.iter().zip(b).fold(0.0, |acc, (a_element, b_element)| {
        acc + (a_element * b_element)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_product_and_sum_equal_lengths() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0];
        let result = dot_product_and_sum(&a, &b);

        assert_eq!(result, 5.0);
    }

    #[test]
    fn dot_product_and_sum_a_bigger_than_b() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0];
        let result = dot_product_and_sum(&a, &b);

        assert_eq!(result, 5.0);
    }

    #[test]
    fn dot_product_and_sum_b_bigger_than_a() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        let result = dot_product_and_sum(&a, &b);

        assert_eq!(result, 5.0);
    }

    #[test]
    fn dot_product_and_sum_a_is_empty_b_is_filled() {
        let a: Vec<f64> = vec![];
        let b = vec![1.0, 2.0];
        let result = dot_product_and_sum(&a, &b);

        assert_eq!(result, 0.0);
    }

    #[test]
    fn dot_product_and_sum_b_is_empty_a_is_filled() {
        let a = vec![1.0, 2.0];
        let b: Vec<f64> = vec![];
        let result = dot_product_and_sum(&a, &b);

        assert_eq!(result, 0.0);
    }
}
