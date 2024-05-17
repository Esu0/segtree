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
}
