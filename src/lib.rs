//! Introduces matrix and vector operations.
//!
//! This implementation does not aim to be fast or optimized but can perfectly
//! be used to compute uniforms for graphic applications.
//! ```
//! use scalararray::MatrixOps;
//!
//! let m1 = [
//!     [1, 2, 3],
//!     [4, 5, 6],
//! ].matrix_transpose();
//!
//! let m2 = [
//!     [0, 1],
//!     [2, 3],
//!     [4, 5],
//! ].matrix_transpose();
//!
//! let m3 = m1.matrix_mul(m2);
//!
//! let m4 = [
//!     [16, 22],
//!     [34, 49],
//! ].matrix_transpose();
//!
//! assert_eq!(m3, m4);
//! ```
//! As matrices are considered to be column major (array of column), it is possible
//! to write and read them in line major by using transposition to perform conversion.

use std::{
    iter::Sum,
    ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign},
};

/// Extends array with matrix operations.
///
/// This implementation consider the matrix to be column major, ie, an array of columns.
/// Implemented for `[[T; M]; N]`, M for the line number and N for the column number.
pub trait MatrixOps<T, const M: usize, const N: usize> {
    /// Transforms all element of the matrix with the given function.
    ///
    /// ```
    /// # use scalararray::MatrixOps;
    /// let m1 = [
    ///     [0, 1, 2],
    ///     [3, 4, 5],
    /// ];
    /// let m2 = m1.matrix_map(|v| if v < 2 { -1 } else { 2 });
    ///
    /// assert_eq!(m2, [
    ///     [-1, -1, 2],
    ///     [ 2,  2, 2],
    /// ]);
    /// ```
    #[must_use]
    fn matrix_map<U, F: FnMut(T) -> U>(self, f: F) -> [[U; M]; N];
    /// Transforms all element of the matrix with the given function and current index.
    ///
    /// ```
    /// # use scalararray::MatrixOps;
    /// let m1 = [
    ///     [10, 13],
    ///     [10, 11],
    /// ];
    /// let m2 = m1.matrix_map_index(|v, m, n| (v, m, n));
    ///
    /// assert_eq!(m2, [
    ///     [(10, 0, 0), (13, 1, 0)],
    ///     [(10, 0, 1), (11, 1, 1)],
    /// ]);
    /// ```
    #[must_use]
    fn matrix_map_index<U, F: FnMut(T, usize, usize) -> U>(self, f: F) -> [[U; M]; N];
    /// Returns the transposed matrix.
    ///
    /// ```
    /// # use scalararray::MatrixOps;
    /// let m1 = [
    ///     [1, 2],
    ///     [3, 4],
    ///     [5, 6],
    /// ];
    /// let m2 = m1.matrix_transpose();
    /// assert_eq!(m2, [
    ///     [1, 3, 5],
    ///     [2, 4, 6],
    /// ]);
    /// ```
    #[must_use]
    fn matrix_transpose(self) -> [[T; N]; M];
    /// Returns the scaled matrix by a given factor.
    ///
    /// ```
    /// # use scalararray::MatrixOps;
    /// let m1 = [
    ///     [1, 2, 3],
    ///     [4, 5, 6],
    /// ];
    /// let m2 = m1.matrix_scale(3);
    /// assert_eq!(m2, [
    ///     [ 3,  6,  9],
    ///     [12, 15, 18],
    /// ]);
    /// ```
    #[must_use]
    fn matrix_scale(self, scalar: T) -> [[T; M]; N]
    where
        T: Mul<T, Output = T>;
    /// Returns the addition of the two matrices.
    ///
    /// ```
    /// # use scalararray::MatrixOps;
    /// let m1 = [
    ///     [1, 2, 3],
    ///     [4, 5, 6],
    /// ];
    /// let m2 = [
    ///     [60, 50, 40],
    ///     [30, 20, 10],
    /// ];
    /// let m3 = m1.matrix_add(m2);
    /// assert_eq!(m3, [
    ///     [61, 52, 43],
    ///     [34, 25, 16],
    /// ]);
    /// ```
    #[must_use]
    fn matrix_add(self, rhs: [[T; M]; N]) -> [[T; M]; N]
    where
        T: Add<T, Output = T>;
    /// Returns the subtraction of the two matrices.
    ///
    /// ```
    /// # use scalararray::MatrixOps;
    /// let m1 = [
    ///     [61, 52, 43],
    ///     [34, 25, 16],
    /// ];
    /// let m2 = [
    ///     [1, 2, 3],
    ///     [4, 5, 6],
    /// ];
    /// let m3 = m1.matrix_sub(m2);
    /// assert_eq!(m3, [
    ///     [60, 50, 40],
    ///     [30, 20, 10],
    /// ]);
    /// ```
    #[must_use]
    fn matrix_sub(self, rhs: [[T; M]; N]) -> [[T; M]; N]
    where
        T: Sub<T, Output = T>;
    /// Returns the multiplication of the two matrices.
    ///
    /// ```
    /// # use scalararray::MatrixOps;
    /// let m1 = [
    ///     [1, 2, 3],
    ///     [4, 5, 6],
    /// ];
    /// let m2 = [
    ///     [0, 1],
    ///     [2, 3],
    ///     [4, 5],
    /// ];
    /// let m3 = m1.matrix_mul(m2);
    /// assert_eq!(m3, [
    ///     [ 4,  5,  6],
    ///     [14, 19, 24],
    ///     [24, 33, 42],
    /// ]);
    /// ```
    #[must_use]
    fn matrix_mul<const O: usize>(self, rhs: [[T; N]; O]) -> [[T; M]; O]
    where
        T: Mul<T, Output = T>,
        T: Sum;
}

impl<T, const M: usize, const N: usize> MatrixOps<T, M, N> for [[T; M]; N]
where
    T: Copy,
{
    fn matrix_map<U, F: FnMut(T) -> U>(self, mut f: F) -> [[U; M]; N] {
        self.map(|col| col.map(&mut f))
    }

    fn matrix_map_index<U, F: FnMut(T, usize, usize) -> U>(self, mut f: F) -> [[U; M]; N] {
        let mut n = 0..N;
        self.map(|col| {
            let mut m = 0..M;
            let n = n.next().unwrap();
            col.map(|elem| f(elem, m.next().unwrap(), n))
        })
    }

    fn matrix_transpose(self) -> [[T; N]; M] {
        let mut option = self.matrix_map(Some);
        [[(); N]; M].matrix_map_index(|(), n, m| option[n][m].take().unwrap())
    }

    fn matrix_add(self, rhs: [[T; M]; N]) -> [[T; M]; N]
    where
        T: Add<T, Output = T>,
    {
        let mut rhs = rhs.into_iter();
        self.map(|lhs| {
            let mut rhs = rhs.next().unwrap().into_iter();
            lhs.map(|lhs| lhs + rhs.next().unwrap())
        })
    }

    fn matrix_sub(self, rhs: [[T; M]; N]) -> [[T; M]; N]
    where
        T: Sub<T, Output = T>,
    {
        let mut rhs = rhs.into_iter();
        self.map(|lhs| {
            let mut rhs = rhs.next().unwrap().into_iter();
            lhs.map(|lhs| lhs - rhs.next().unwrap())
        })
    }

    fn matrix_mul<const O: usize>(self, rhs: [[T; N]; O]) -> [[T; M]; O]
    where
        T: Mul<T, Output = T>,
        T: Sum,
    {
        [[(); M]; O].matrix_map_index(|_, m, o| (0..N).map(|n| self[n][m] * rhs[o][n]).sum())
    }

    fn matrix_scale(self, scalar: T) -> [[T; M]; N]
    where
        T: Mul<T, Output = T>,
    {
        self.matrix_map(|v| v * scalar)
    }
}

/// Extends array with vector operations.
pub trait VectorOps<T, const N: usize> {
    /// Equivalent to array map.
    #[must_use]
    fn vector_map<U>(self, f: impl FnMut(T) -> U) -> [U; N];
    /// Returns the mapped vector with the given function and the current index.
    #[must_use]
    fn vector_map_index<U>(self, f: impl FnMut(T, usize) -> U) -> [U; N];

    /// Returns the scalled vector by a given factor.
    #[must_use]
    fn vector_scale(self, scalar: T) -> [T; N]
    where
        T: Mul<T, Output = T>;

    /// Returns the negative of the vector.
    #[must_use]
    fn vector_neg(self) -> [T; N]
    where
        T: Neg<Output = T>;

    /// Returns the addition of the two vector.
    #[must_use]
    fn vector_add(self, rhs: [T; N]) -> [T; N]
    where
        T: Add<T, Output = T>;

    /// Performs add assignation.
    fn vector_add_assign(&mut self, rhs: [T; N])
    where
        T: AddAssign<T>;

    /// Returns the subtraction of the two vector.
    #[must_use]
    fn vector_sub(self, rhs: [T; N]) -> [T; N]
    where
        T: Sub<T, Output = T>;

    /// Performs the subtraction assignation.
    fn vector_sub_assign(&mut self, rhs: [T; N])
    where
        T: SubAssign<T>;

    /// Returns the dot product of the two vector.
    #[must_use]
    fn vector_dot(self, rhs: [T; N]) -> T
    where
        T: Mul<T, Output = T>,
        T: Sum;
}

impl<T, const N: usize> VectorOps<T, N> for [T; N]
where
    T: Copy,
{
    fn vector_map<U>(self, f: impl FnMut(T) -> U) -> [U; N] {
        self.map(f)
    }

    fn vector_scale(self, scalar: T) -> [T; N]
    where
        T: Mul<T, Output = T>,
    {
        self.map(|v| v * scalar)
    }

    fn vector_neg(self) -> [T; N]
    where
        T: Neg<Output = T>,
    {
        self.map(|v| v.neg())
    }

    fn vector_add(self, rhs: [T; N]) -> [T; N]
    where
        T: Add<T, Output = T>,
    {
        let mut rhs = rhs.into_iter();
        self.map(|lhs| lhs + rhs.next().unwrap())
    }

    fn vector_add_assign(&mut self, rhs: [T; N])
    where
        T: AddAssign<T>,
    {
        for (dst, src) in self.iter_mut().zip(rhs.into_iter()) {
            *dst += src;
        }
    }

    fn vector_sub(self, rhs: [T; N]) -> [T; N]
    where
        T: Sub<T, Output = T>,
    {
        let mut rhs = rhs.into_iter();
        self.map(|lhs| lhs - rhs.next().unwrap())
    }

    fn vector_sub_assign(&mut self, rhs: [T; N])
    where
        T: SubAssign<T>,
    {
        for (dst, src) in self.iter_mut().zip(rhs.into_iter()) {
            *dst -= src;
        }
    }

    fn vector_dot(self, rhs: [T; N]) -> T
    where
        T: Mul<T, Output = T>,
        T: Sum,
    {
        self.into_iter()
            .zip(rhs.into_iter())
            .map(|(lhs, rhs)| lhs * rhs)
            .sum()
    }

    fn vector_map_index<U>(self, mut f: impl FnMut(T, usize) -> U) -> [U; N] {
        let mut index = 0..;
        self.map(|v| f(v, index.next().unwrap()))
    }
}

pub trait Rotation<T, const N: usize> {
    fn rotation_x(radian: T) -> [[T; N]; N];
    fn rotation_y(radian: T) -> [[T; N]; N];
    fn rotation_z(radian: T) -> [[T; N]; N];
}

impl Rotation<f64, 3> for [[f64; 3]; 3] {
    fn rotation_x(radian: f64) -> [[f64; 3]; 3] {
        let y_y = radian.cos();
        let z_z = radian.cos();
        let y_z = -radian.sin();
        let z_y = radian.sin();
        [[1.0, 0.0, 0.0], [0.0, y_y, y_z], [0.0, z_y, z_z]]
    }

    fn rotation_y(radian: f64) -> [[f64; 3]; 3] {
        let x_x = radian.cos();
        let z_z = radian.cos();
        let x_z = radian.sin();
        let z_x = -radian.sin();
        [[x_x, 0.0, x_z], [0.0, 1.0, 0.0], [z_x, 0.0, z_z]]
    }

    fn rotation_z(radian: f64) -> [[f64; 3]; 3] {
        let x_x = radian.cos();
        let y_y = radian.cos();
        let x_y = -radian.sin();
        let y_x = radian.sin();
        [[x_x, x_y, 0.0], [y_x, y_y, 0.0], [0.0, 0.0, 1.0]]
    }
}

pub trait VectorRotateOps<T, const N: usize> {
    #[must_use]
    fn vector_rotate_x(self, radian: T) -> [T; N];
    #[must_use]
    fn vector_rotate_y(self, radian: T) -> [T; N];
    #[must_use]
    fn vector_rotate_z(self, radian: T) -> [T; N];
}

impl VectorRotateOps<f64, 3> for [f64; 3] {
    fn vector_rotate_x(self, radian: f64) -> [f64; 3] {
        <[[f64; 3]; 3]>::rotation_x(radian).matrix_mul([self])[0]
    }

    fn vector_rotate_y(self, radian: f64) -> [f64; 3] {
        <[[f64; 3]; 3]>::rotation_y(radian).matrix_mul([self])[0]
    }

    fn vector_rotate_z(self, radian: f64) -> [f64; 3] {
        <[[f64; 3]; 3]>::rotation_z(radian).matrix_mul([self])[0]
    }
}
