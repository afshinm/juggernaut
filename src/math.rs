use std::ops::Mul;


/// Vector multiplication
trait Multiplication<T> {
    fn dot(&self, x: T) -> Self;
}

impl<T> Multiplication<T> for Vec<T>
where
    T: Mul<Output = T> + Copy,
{
    fn dot(&self, x: T) -> Vec<T> {
        self.iter().map(|y| *y * x).collect()
    }
}

#[cfg(test)]
mod tests {
    use math::Multiplication;

    #[test]
    fn vec_dot() {
        assert_eq!(vec![1, 2, 3].dot(2), vec![2, 4, 6]);
        assert_eq!(vec![5, 0, 1].dot(5), vec![25, 0, 5]);
    }

}
