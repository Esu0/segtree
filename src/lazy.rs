use std::{borrow::Borrow, ops::{Add, Deref, Range}};

use crate::{
    query::{ident::HasAddIdent, MinQuery, Query, SumQuery},
    SegTree,
};

#[derive(Debug)]
pub struct LazySegTree<Q, O, T, U> {
    operator: O,
    lazy: Box<[U]>,
    base: SegTree<Q, T>,
}

enum Cow<'a, T> {
    Borrowed(&'a T),
    Owned(T),
}

impl<'a, T> Cow<'a, T> {
    pub fn into_owned_with(self, f: impl FnOnce(&T) -> T) -> T {
        match self {
            Cow::Borrowed(b) => f(b),
            Cow::Owned(o) => o,
        }
    }
}

impl<'a, T> Deref for Cow<'a, T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        match self {
            Cow::Borrowed(b) => *b,
            Cow::Owned(b) => b,
        }
    }
}

impl<T, U, Q: Query<T>, O: LeftOperator<Q, T, U>> LazySegTree<Q, O, T, U> {
    pub fn from_iter_query_operater<I>(query: Q, operator: O, iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        let base = SegTree::from_iter_query(query, iter);
        let n = base.len();
        let lazy = std::iter::repeat_with(|| O::IDENT)
            .take(2 * n)
            .collect::<Box<[_]>>();
        Self {
            operator,
            lazy,
            base,
        }
    }

    pub fn len(&self) -> usize {
        self.base.len()
    }

    pub fn is_empty(&self) -> bool {
        self.base.is_empty()
    }

    pub fn update(&mut self, range: Range<usize>, value: U) {
        let (l, r) = self.base.get_lr(range);
        self.update_rec(l, r, 0, self.len(), 1, &value);
    }

    fn update_rec(&mut self, a: usize, b: usize, l: usize, r: usize, i: usize, value: &U) {
        if r <= a || b <= l {
            return;
        }
        if a <= l && r <= b {
            self.lazy[i] = self.operator.apply_to_operator(&self.lazy[i], value);
            self.base.tree[i] = self.operator.apply_n(&self.base.tree[i], value, r - l);
            return;
        }
        let mid = (l + r) / 2;
        let ch1 = 2 * i;
        let ch2 = ch1 + 1;
        self.base.tree[ch1] = self
            .operator
            .apply_n(&self.base.tree[ch1], &self.lazy[i], mid - l);
        self.lazy[ch1] = self
            .operator
            .apply_to_operator(&self.lazy[ch1], &self.lazy[i]);
        self.update_rec(a, b, l, mid, ch1, value);
        self.base.tree[ch2] = self
            .operator
            .apply_n(&self.base.tree[ch2], &self.lazy[i], r - mid);
        self.lazy[ch2] = self
            .operator
            .apply_to_operator(&self.lazy[ch2], &self.lazy[i]);
        self.update_rec(a, b, mid, r, ch2, value);
        self.base.tree[i] = self
            .base
            .query
            .query(&self.base.tree[ch1], &self.base.tree[ch2]);
        self.lazy[i] = O::IDENT;
    }

    // pub fn query(&mut self, range: Range<usize>) -> T {
    //     let (l, r) = self.base.get_lr(range);
    //     self.query_rec(l, r, 0, self.len(), 1).into_owned_with(|x| self.base.query.query(x, &Q::IDENT))
    // }

    // fn query_rec(&mut self, a: usize, b: usize, l: usize, r: usize, i: usize) -> Cow<'_, T>
    // where
    //     T: ToOwned<Owned = T>,
    // {
    //     if r <= a || b <= l {
    //         Cow::Borrowed(&Q::IDENT)
    //     } else if a <= l && r <= b {
    //         Cow::Borrowed(&self.base.tree[i])
    //     } else {
    //         let mid = (l + r) / 2;
    //         let ch1 = 2 * i;
    //         let ch2 = ch1 + 1;
    //         self.base.tree[ch1] = self
    //             .operator
    //             .apply_n(&self.base.tree[ch1], &self.lazy[i], mid - l);

    //     }
    // }
}

pub struct Update;

impl<T> LeftOperator<MinQuery, T, Option<T>> for Update
where
    MinQuery: Query<T>,
    T: Clone,
{
    const IDENT: Option<T> = None;
    fn apply_n(&self, x: &T, y: &Option<T>, _: usize) -> T {
        match y {
            Some(y) => y.clone(),
            None => x.clone(),
        }
    }

    fn apply_to_operator(&self, x: &Option<T>, y: &Option<T>) -> Option<T> {
        y.as_ref().or(x.as_ref()).cloned()
    }
}

pub trait Times {
    type Output;

    fn times(&self, n: usize) -> Self::Output;
}

impl Times for u32 {
    type Output = u32;
    fn times(&self, n: usize) -> Self::Output {
        self * n as u32
    }
}

#[derive(Clone, Copy, Debug)]
pub struct AddOperator;

impl<T, U> LeftOperator<SumQuery, T, U> for AddOperator
where
    SumQuery: Query<T>,
    T: Clone + Add<<U as Times>::Output, Output = T>,
    U: Clone + Times + Add<U, Output = U> + HasAddIdent,
{
    const IDENT: U = U::IDENT;
    fn apply_n(&self, x: &T, y: &U, n: usize) -> T {
        x.clone() + y.times(n)
    }

    fn apply_to_operator(&self, x: &U, y: &U) -> U {
        x.clone() + y.clone()
    }
}

pub trait LeftOperator<Q, T, U>
where
    Q: Query<T>,
{
    const IDENT: U;
    /// `x`に`y`を作用させた結果を返す。
    ///
    /// # Note
    /// `x`はある`x_1, x_2, ..., x_n`について、`query(x_1, query(x_2, ..., query(x_{n-1}, x_n)))`の結果であり、
    /// この関数は`query(apply(x_1, y), query(apply(x_2, y), ..., query(apply(x_{n-1}, y), apply(x_n, y))))`を返すべきであることに注意する。
    fn apply_n(&self, x: &T, y: &U, n: usize) -> T;

    fn apply_to_operator(&self, x: &U, y: &U) -> U;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lazy_segtree_update() {
        let mut segtree =
            LazySegTree::<_, _, _, u32>::from_iter_query_operater(SumQuery, AddOperator, 1u32..=10);
        segtree.update(0..5, 3);
        segtree.update(1..5, 2);
        segtree.update(3..9, 1);
        println!("{:?}", segtree.base.tree);
        println!("{:?}", segtree.lazy);
        segtree.update(5..10, 4);
        println!("{:?}", segtree.base.tree);
        println!("{:?}", segtree.lazy);
    }
}
