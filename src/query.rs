use std::{
    cmp::Ordering,
    ops::{Add, Mul, Rem},
};

use self::ident::{HasAddIdent, HasMax, HasMin, HasMulIdent};

pub mod ident;

pub trait Query<T> {
    const IDENT: T;
    fn query(&self, x: &T, y: &T) -> T;
}

pub trait Additional<T> {
    type Ret;
    const IDENT: Self::Ret;
    fn extract(&self, slice: &[T]) -> Self::Ret;
}

pub trait QueryWith<T> {
    const IDENT: T;
    type A: Additional<T>;
    fn additional(&self) -> Self::A;
    fn query_with(
        &self,
        x: &T,
        y: &T,
        additional_x: <Self::A as Additional<T>>::Ret,
        additional_y: <Self::A as Additional<T>>::Ret,
    ) -> (T, <Self::A as Additional<T>>::Ret);
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NoAdditional;

impl<T> Additional<T> for NoAdditional {
    type Ret = ();
    const IDENT: Self::Ret = ();
    fn extract(&self, _: &[T]) {}
}

// impl<Q, T> QueryWith<T> for Q
// where
//     Q: Query<T>,
// {
//     const IDENT: T = Q::IDENT;
//     type A = NoAdditional;
//     fn query_with(&self, x: &T, y: &T, _: (), _: ()) -> (T, ()) {
//         (self.query(x, y), ())
//     }
// }

macro_rules! impl_query_with_where_query {
    ($($t:ty),* $(,)?) => {
        $(impl<T> QueryWith<T> for $t
        where
            $t: Query<T>,
        {
            const IDENT: T = <$t as Query<T>>::IDENT;
            type A = NoAdditional;
            fn additional(&self) -> Self::A {
                NoAdditional
            }
            fn query_with(&self, x: &T, y: &T, _: (), _: ()) -> (T, ()) {
                (self.query(x, y), ())
            }
        })*
    }
}

impl_query_with_where_query!(MinQuery, MaxQuery, SumQuery, ProdQuery, GcdQuery);

#[derive(Clone, Copy, Debug, Default)]
pub struct MinQuery;

impl<T: Ord + HasMax + Clone> Query<T> for MinQuery {
    const IDENT: T = T::MAX;
    fn query(&self, x: &T, y: &T) -> T {
        match x.cmp(y) {
            Ordering::Less => x.clone(),
            _ => y.clone(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct MaxQuery;

impl<T: Ord + HasMin + Clone> Query<T> for MaxQuery {
    const IDENT: T = T::MIN;
    fn query(&self, x: &T, y: &T) -> T {
        match x.cmp(y) {
            Ordering::Less => y.clone(),
            _ => x.clone(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SumQuery;

impl<T: Add<Output = T> + HasAddIdent + Clone> Query<T> for SumQuery {
    const IDENT: T = T::IDENT;
    fn query(&self, x: &T, y: &T) -> T {
        x.clone() + y.clone()
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ProdQuery;

impl<T: Mul<Output = T> + HasMulIdent + Clone> Query<T> for ProdQuery {
    const IDENT: T = T::IDENT;
    fn query(&self, x: &T, y: &T) -> T {
        x.clone() * y.clone()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Mod<Q, T> {
    base: Q,
    modulo: T,
}

impl<Q, T: Rem<Output = T>> Mod<Q, T> {
    pub fn new(query: Q, modulo: T) -> Self {
        Self {
            base: query,
            modulo,
        }
    }
}

impl<Q, T, U> Query<U> for Mod<Q, T>
where
    Q: Query<U>,
    T: Clone,
    U: Clone + Rem<T, Output = U>,
{
    const IDENT: U = Q::IDENT;
    fn query(&self, x: &U, y: &U) -> U {
        self.base.query(x, y) % self.modulo.clone()
    }
}

impl<Q, T, U> QueryWith<U> for Mod<Q, T>
where
    Mod<Q, T>: Query<U>,
{
    const IDENT: U = <Mod<Q, T> as Query<U>>::IDENT;
    type A = NoAdditional;
    fn additional(&self) -> Self::A {
        NoAdditional
    }
    fn query_with(&self, x: &U, y: &U, _: (), _: ()) -> (U, ()) {
        (self.query(x, y), ())
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct GcdQuery;

impl<T: Rem<Output = T> + HasAddIdent + Eq + Clone> Query<T> for GcdQuery {
    const IDENT: T = T::IDENT;
    fn query(&self, x: &T, y: &T) -> T {
        if x == &T::IDENT {
            return y.clone();
        }
        let mut x = x.clone();
        let mut y = y.clone();
        while y != T::IDENT {
            let tmp = x.clone() % y.clone();
            x = std::mem::replace(&mut y, tmp);
        }
        x
    }
}

#[derive(Clone, Debug)]
pub struct PolynomialQuery<T, SQ = SumQuery, PQ = ProdQuery> {
    exponents: Vec<T>,
    sum_query: SQ,
    prod_query: PQ,
}

impl<SQ: QueryWith<T, A = NoAdditional>, PQ: QueryWith<T, A = NoAdditional>, T>
    PolynomialQuery<T, SQ, PQ>
{
    pub fn new(exp: T, n: usize) -> Self
    where
        SQ: Default,
        PQ: Default,
    {
        Self::with_query(exp, n, SQ::default(), PQ::default())
    }

    pub fn with_query(exp: T, n: usize, sum_query: SQ, prod_query: PQ) -> Self {
        if n == 0 {
            Self {
                exponents: vec![],
                prod_query,
                sum_query,
            }
        } else if n == 1 {
            Self {
                exponents: vec![PQ::IDENT],
                prod_query,
                sum_query,
            }
        } else {
            let mut exponents = Vec::with_capacity(n);
            exponents.extend([PQ::IDENT, exp]);
            for _ in 2..n {
                let tmp = prod_query
                    .query_with(exponents.last().unwrap(), &exponents[1], (), ())
                    .0;
                exponents.push(tmp);
            }
            Self {
                exponents,
                prod_query,
                sum_query,
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct WithLen;

impl<T> Additional<T> for WithLen {
    type Ret = usize;
    const IDENT: Self::Ret = 0;
    fn extract(&self, slice: &[T]) -> Self::Ret {
        slice.len()
    }
}

impl<SQ, PQ, T> QueryWith<T> for PolynomialQuery<T, SQ, PQ>
where
    SQ: QueryWith<T, A = NoAdditional>,
    PQ: QueryWith<T, A = NoAdditional>,
{
    const IDENT: T = SQ::IDENT;
    type A = WithLen;
    fn additional(&self) -> Self::A {
        WithLen
    }
    fn query_with(&self, x: &T, y: &T, degree_x: usize, degree_y: usize) -> (T, usize) {
        (
            self.sum_query
                .query_with(
                    x,
                    &self
                        .prod_query
                        .query_with(y, &self.exponents[degree_x], (), ())
                        .0,
                    (),
                    (),
                )
                .0,
            degree_x + degree_y,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mod_query() {
        let query = Mod::new(SumQuery, 1000);
        assert_eq!(query.query(&1, &2), 3);
        assert_eq!(query.query(&999, &999), 998);
        assert_eq!(query.query(&999, &2), 1);

        let query = Mod::new(ProdQuery, 1000);
        assert_eq!(query.query(&2, &3), 6);
        assert_eq!(query.query(&999, &999), 1);
        assert_eq!(query.query(&999, &2), 998);
        assert_eq!(query.query(&50, &50), 500);
    }

    #[test]
    fn test_gcd_query() {
        assert_eq!(GcdQuery.query(&10, &15), 5);
        assert_eq!(GcdQuery.query(&10, &10), 10);
        assert_eq!(GcdQuery.query(&10, &5), 5);
        assert_eq!(GcdQuery.query(&10, &3), 1);
        assert_eq!(GcdQuery.query(&0, &4), 4);
        assert_eq!(GcdQuery.query(&4, &0), 4);
    }

    #[test]
    fn test_polynomial_query() {
        let query = PolynomialQuery::with_query(3, 8, SumQuery, ProdQuery);
        assert_eq!(&query.exponents, &[1, 3, 9, 27, 81, 243, 729, 2187]);

        // 1 * 3^0 + 2 * 3^1
        assert_eq!(query.query_with(&1, &2, 1, 1), (7, 2));
        // 1 * 3^0 + 2 * 3^1 - 2 * 3^2
        assert_eq!(query.query_with(&7, &-2, 2, 1), (-11, 3));
        // 2 * 3^0 - 2 * 3^1
        assert_eq!(query.query_with(&2, &-2, 1, 1), (-4, 2));
        // 1 * 3^0 + 2 * 3^1 - 2 * 3^2
        assert_eq!(query.query_with(&1, &-4, 1, 2), (-11, 3));

        let result = [1i32, 2, 3, 4, 5, 4]
            .into_iter()
            .map(|x| (x, 1))
            .reduce(|acc, x| query.query_with(&acc.0, &x.0, acc.1, x.1))
            .unwrap();
        assert_eq!(result, (1519, 6));

        let query =
            PolynomialQuery::with_query(3, 8, Mod::new(SumQuery, 221), Mod::new(ProdQuery, 221));
        assert_eq!(
            &query.exponents,
            &[1, 3, 9, 27, 81, 243 % 221, 729 % 221, 2187 % 221]
        );

        let result = [1i32, 2, 3, 4, 5, 4]
            .into_iter()
            .map(|x| (x, 1))
            .reduce(|acc, x| query.query_with(&acc.0, &x.0, acc.1, x.1))
            .unwrap();
        assert_eq!(result, (1519 % 221, 6));
    }
}
