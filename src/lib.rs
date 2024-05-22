pub mod query;

use std::{
    alloc::Layout, borrow::Borrow, cmp::Ordering, mem::MaybeUninit, ops::{Bound, RangeBounds}
};

use query::{Additional, MinQuery, Query, QueryWith};

#[derive(Clone, Debug)]
pub struct SegTree<Q: QueryWith<T>, T> {
    tree: Box<[T]>,
    query: Q,
    additional: Q::A,
}

impl<T, Q: QueryWith<T>> SegTree<Q, T> {
    /// half_len != 0
    unsafe fn make_tree_ptr(
        half_len: usize,
        f: impl FnOnce(*mut T) -> usize,
        query: &Q,
        additional: &Q::A,
    ) -> *mut [T] {
        let len = half_len * 2;
        let ptr = std::alloc::alloc(Layout::array::<T>(len).unwrap()) as *mut T;
        {
            let data_ptr = ptr.add(half_len);
            let orig_len = f(data_ptr);
            for i in orig_len..half_len {
                data_ptr.add(i).write(Q::IDENT);
            }
        }
        ptr.write(Q::IDENT);
        Self::eval(ptr, half_len, query, additional);

        std::ptr::slice_from_raw_parts_mut(ptr, len)
    }

    unsafe fn from_write_fn(
        half_len: usize,
        query: Q,
        additional: Q::A,
        f: impl FnOnce(*mut T) -> usize,
    ) -> Self {
        Self {
            tree: Box::from_raw(Self::make_tree_ptr(half_len, f, &query, &additional)),
            query,
            additional,
        }
    }

    fn new_empty(query: Q) -> Self {
        Self {
            tree: Box::new([]),
            additional: query.additional(),
            query,
        }
    }

    /// データのスライスからセグメント木を構築する。
    pub fn new(query: Q, data: &[T]) -> Self
    where
        T: Clone,
    {
        let orig_len = data.len();
        if orig_len == 0 {
            return Self::new_empty(query);
        }
        let half_len = orig_len.next_power_of_two();
        let additional = query.additional();
        unsafe {
            Self::from_write_fn(half_len, query, additional, |data_ptr| {
                for (i, data_i) in data.iter().enumerate() {
                    data_ptr.add(i).write(data_i.clone())
                }
                orig_len
            })
        }
    }

    pub fn from_iter_query<I>(query: Q, iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        let iter = iter.into_iter();
        let (size_min, size_max) = iter.size_hint();
        let additional = query.additional();
        if size_max == Some(0) {
            Self::new_empty(query)
        } else {
            assert_ne!(size_min, 0);
            let half_len_min = size_min.next_power_of_two();
            let half_len_max = size_max.map(usize::next_power_of_two);
            if Some(half_len_min) == half_len_max {
                let half_len = half_len_min;
                unsafe {
                    Self::from_write_fn(half_len, query, additional, move |data_ptr| {
                        let mut i = 0;
                        for item in iter {
                            data_ptr.add(i).write(item);
                            i += 1;
                        }
                        i
                    })
                }
            } else {
                let mut data = iter.collect::<Vec<_>>();
                let orig_len = data.len();
                unsafe {
                    Self::from_write_fn(
                        orig_len.next_power_of_two(),
                        query,
                        additional,
                        move |data_ptr| {
                            let src = data.as_mut_ptr();
                            let cap = data.capacity();
                            std::mem::forget(data);
                            data_ptr.copy_from_nonoverlapping(src, orig_len);
                            // `I`のデストラクタは呼ばずにメモリの解放のみ行う
                            drop(Vec::from_raw_parts(
                                src as *mut MaybeUninit<I>,
                                orig_len,
                                cap,
                            ));
                            orig_len
                        },
                    )
                }
            }
        }
    }

    pub fn len(&self) -> usize {
        self.tree.len() / 2
    }

    pub fn is_empty(&self) -> bool {
        self.tree.is_empty()
    }

    unsafe fn eval(ptr: *mut T, half_len: usize, query: &Q, additional: &Q::A) {
        let len = half_len * 2;
        let mut range_end = len;
        let mut range_len = 1;
        for i in (1..half_len).rev() {
            range_end -= range_len * 2;
            let slice1 = std::ptr::slice_from_raw_parts(ptr.add(range_end + range_len), range_len);
            let slice2 = std::ptr::slice_from_raw_parts(ptr.add(range_end), range_len);
            ptr.add(i).write(
                query
                    .query_with(
                        &*ptr.add(i * 2),
                        &*ptr.add(i * 2 + 1),
                        additional.additional(&*slice1),
                        additional.additional(&*slice2),
                    )
                    .0,
            );
            if range_end <= half_len {
                range_end = len;
                range_len <<= 1;
            }
        }
    }

    /// 戻り値を`(l, r)`とすると以下が保証される。
    ///
    /// * `l <= r <= self.len()`
    fn get_lr(&self, range: impl RangeBounds<usize>) -> (usize, usize) {
        let size = self.len();
        let l = match range.start_bound() {
            Bound::Excluded(s) => s
                .checked_add(1)
                .unwrap_or_else(|| panic!("attempted to index slice from after maximum usize")),
            Bound::Included(s) => *s,
            Bound::Unbounded => 0,
        };
        let r = match range.end_bound() {
            Bound::Excluded(e) => *e,
            Bound::Included(e) => e
                .checked_add(1)
                .unwrap_or_else(|| panic!("attempted to index slice up to maximum usize")),
            Bound::Unbounded => size,
        };
        if l > r {
            panic!("slice index starts at {l} but ends at {r}");
        } else if r > size {
            panic!("range end index {r} out of range for slice of length {size}");
        }
        (l, r)
    }

    /// 指定区間のクエリをO(log(n))で求める。
    pub fn query(&self, range: impl RangeBounds<usize>) -> T {
        let (mut l, mut r) = self.get_lr(range);
        if r == l {
            return Q::IDENT;
        }
        l += self.len();
        r += self.len();
        let mut l_query = Q::IDENT;
        let mut l_query_a = Q::A::IDENT;
        let mut r_query = Q::IDENT;
        let mut r_query_a = Q::A::IDENT;

        let mut l_range_start = l;
        // let mut l_range_end = l + 1;
        // let mut r_range_start = r - 1;
        let mut r_range_end = r;
        let mut arr_len = 1usize;
        while r - l > 2 {
            if l & 1 == 1 {
                let l_range_end = l_range_start + arr_len;
                let ret = unsafe { self.query.query_with(
                    &l_query,
                    &self.tree[l],
                    l_query_a,
                    self.additional
                        .additional(self.tree.get_unchecked(l_range_start..l_range_end)),
                )};
                l_query = ret.0;
                l_query_a = ret.1;
                l += 1;
                l_range_start = l_range_end;
            }
            if r & 1 == 1 {
                r -= 1;
                let r_range_start = r_range_end - arr_len;
                let ret = unsafe {self.query.query_with(
                    &self.tree[r],
                    &r_query,
                    self.additional
                        .additional(self.tree.get_unchecked(r_range_start..r_range_end)),
                    r_query_a,
                )};
                r_query = ret.0;
                r_query_a = ret.1;
                r_range_end = r_range_start;
            }
            arr_len <<= 1;
            l >>= 1;
            r >>= 1;
        }
        let a = unsafe { self.query.query_with(
            &l_query,
            &self.tree[l],
            l_query_a,
            self.additional
                .additional(self.tree.get_unchecked(l_range_start..l_range_start + arr_len)),
        )};
        if r - l == 2 {
            // [&self.tree[l], &self.tree[l + 1], &r_query]
            //     .into_iter()
            //     .fold(l_query, |acc, x| self.query.query(&acc, x))
            let b = unsafe {self.query.query_with(
                &a.0,
                &self.tree[l + 1],
                a.1,
                self.additional
                    .additional(self.tree.get_unchecked(r_range_end - arr_len..r_range_end)),
            )};
            self.query.query_with(&b.0, &r_query, b.1, r_query_a).0
        } else {
            // [&self.tree[l], &r_query]
            //     .into_iter()
            //     .fold(l_query, |acc, x| self.query.query(&acc, x))
            self.query.query_with(&a.0, &r_query, a.1, r_query_a).0
        }
    }

    /// 指定位置の要素をO(log(n))で更新する。
    pub fn update(&mut self, i: usize, val: T) {
        let mut i = i
            .checked_add(self.len())
            .unwrap_or_else(|| panic!("attempt to index slice maximum usize"));
        self.tree[i] = val;
        let mut range_start = i;
        let mut range_len = 1;
        while i > 1 {
            i >>= 1;
            range_start &= !range_len;
            self.tree[i] = self
                .query
                .query_with(
                    &self.tree[i * 2],
                    &self.tree[i * 2 + 1],
                    self.additional
                        .additional(&self.tree[range_start..range_start + range_len]),
                    self.additional.additional(
                        &self.tree[range_start + range_len..range_start + range_len * 2],
                    ),
                )
                .0;
            range_len <<= 1;
        }
    }

    /// `pred(self.query(l..j))`が`true`となる最大の`j`をO(log(n))で求める。
    pub fn partition_point<P>(&self, l: usize, mut pred: P) -> usize
    where
        P: FnMut(&T) -> bool,
        <Q::A as Additional<T>>::Ret: Clone,
    {
        match l.cmp(&self.len()) {
            Ordering::Equal => return l,
            Ordering::Greater => {
                panic!("index {l} out of range for slice of length {}", self.len())
            }
            _ => {}
        }
        let mut l = l
            .checked_add(self.len())
            .unwrap_or_else(|| panic!("attempt to index maximum usize"));
        let mut l_range_start = l;
        let mut arr_len = 1usize;

        let mut l_query = Q::IDENT;
        let mut l_query_a = Q::A::IDENT;
        loop {
            if l & 1 == 1 {
                let next_query = self.query.query_with(
                    &l_query,
                    &self.tree[l],
                    l_query_a.clone(),
                    self.additional
                        .additional(&self.tree[l_range_start..l_range_start + arr_len]),
                );
                let next_l = l + 1;
                if pred(&next_query.0) {
                    if next_l.is_power_of_two() {
                        return self.len();
                    } else {
                        l_query = next_query.0;
                        l_query_a = next_query.1;
                    }
                } else {
                    break;
                }
                l = next_l;
                l_range_start += arr_len;
            }
            arr_len <<= 1;
            l >>= 1;
        }
        loop {
            let next_l = l << 1;
            let Some(val) = self.tree.get(next_l) else {
                return l - self.len();
            };
            l = next_l;
            arr_len >>= 1;
            let next_query = self.query.query_with(
                &l_query,
                val,
                l_query_a.clone(),
                self.additional
                    .additional(&self.tree[l_range_start..l_range_start + arr_len]),
            );
            if pred(&next_query.0) {
                l_query = next_query.0;
                l_query_a = next_query.1;
                l += 1;
            }
        }
    }
}

impl<T> SegTree<MinQuery, T>
where
    MinQuery: Query<T>,
{
    // セグメント木の全要素を'val'で埋める。
    pub fn fill(&mut self, val: T)
    where
        T: Clone,
    {
        self.tree.fill(val);
    }

    /// セグメント木の全要素を`f`の戻り値で埋める。
    ///
    /// # Note
    /// 内部可変性を用いると`f`の戻り値が変化するようにすることもできるが、その場合はセグメント木は意味のない値で埋められ、
    /// クエリの結果も意味のない値になる。
    pub fn fill_with(&mut self, f: impl Fn() -> T) {
        self.tree.fill_with(f)
    }

    /// `k`と等しいかそれより小さい要素の最左位置をO(log(n))で求める。
    ///
    /// self.query(..j) > kとなる最大のjを返す。
    pub fn upper_bound<Q: ?Sized>(&self, k: &Q) -> usize
    where
        T: Ord + Borrow<Q>,
        Q: Ord,
    {
        self.partition_point(0, |v| v.borrow() > k)
    }
}

impl<I, Q: QueryWith<I> + Default> FromIterator<I> for SegTree<Q, I> {
    fn from_iter<T: IntoIterator<Item = I>>(iter: T) -> Self {
        Self::from_iter_query(Q::default(), iter)
    }
}

#[cfg(test)]
mod tests {
    use rand::{rngs::StdRng, Rng, SeedableRng};

    use crate::query::{MaxQuery, Mod, SumQuery};

    use self::query::{PolynomialQuery, ProdQuery};

    use super::*;

    #[test]
    fn new_test() {
        let segtree = SegTree::new(MinQuery, &[1u32, 2, 3, 4, 5, 6]);
        assert_eq!(
            &segtree.tree[1..],
            &[
                1,
                1,
                5,
                1,
                3,
                5,
                u32::MAX,
                1,
                2,
                3,
                4,
                5,
                6,
                u32::MAX,
                u32::MAX
            ]
        );
    }

    #[test]
    fn from_iter_test() {
        let segtree = [100, 200, 15, 40]
            .into_iter()
            .collect::<SegTree<SumQuery, _>>();
        assert_eq!(&segtree.tree[1..], &[355, 300, 55, 100, 200, 15, 40]);
    }

    #[test]
    fn sum_query_test() {
        let segtree = [-4, 6, -3, 2, 1, 1, 7]
            .into_iter()
            .collect::<SegTree<SumQuery, _>>();

        assert_eq!(segtree.query(..), 10);
        assert_eq!(segtree.query(3..), 11);
        assert_eq!(segtree.query(3..6), 4);
        assert_eq!(segtree.query(..3), -1);

        assert_eq!(segtree.query(0..1), -4);
        assert_eq!(segtree.query(0..=0), -4);
        assert_eq!(segtree.query(0..=1), 2);
        assert_eq!(segtree.query(0..0), 0);
        assert_eq!(segtree.query(1..1), 0);
        assert_eq!(segtree.query(7..7), 0);
        assert_eq!(segtree.query(6..8), 7);
    }

    #[test]
    fn min_query_test() {
        let segtree = [23i32, 12, -3, 0, 3, -2, 7, 8]
            .into_iter()
            .collect::<SegTree<MinQuery, _>>();

        assert_eq!(segtree.query(..), -3);
        assert_eq!(segtree.query(3..), -2);
        assert_eq!(segtree.query(..2), 12);
        assert_eq!(segtree.query(3..5), 0);

        assert_eq!(segtree.query(0..1), 23);
        assert_eq!(segtree.query(0..=0), 23);
        assert_eq!(segtree.query(0..=1), 12);
        assert_eq!(segtree.query(0..0), i32::MAX);
        assert_eq!(segtree.query(1..1), i32::MAX);
        assert_eq!(segtree.query(7..7), i32::MAX);
        assert_eq!(segtree.query(7..8), 8);
    }

    #[test]
    #[should_panic]
    fn out_of_bounds_test1() {
        let segtree = [1, 2, 3, 4, 5, 6, 7]
            .into_iter()
            .collect::<SegTree<SumQuery, _>>();
        segtree.query(0..9);
    }

    #[test]
    #[should_panic]
    fn out_of_bounds_test2() {
        let segtree = [1, 2, 3, 4, 5, 6, 7]
            .into_iter()
            .collect::<SegTree<SumQuery, _>>();
        segtree.query(9..);
    }

    #[test]
    #[should_panic]
    fn out_of_bounds_test3() {
        let segtree = [1, 2, 3, 4, 5, 6, 7]
            .into_iter()
            .collect::<SegTree<SumQuery, _>>();
        segtree.query(0..=8);
    }

    #[test]
    #[should_panic]
    #[allow(clippy::reversed_empty_ranges)]
    fn out_of_bounds_test4() {
        let segtree = [1, 2, 3, 4, 5, 6, 7]
            .into_iter()
            .collect::<SegTree<SumQuery, _>>();
        segtree.query(5..4);
    }

    #[test]
    fn update_test() {
        let mut segtree = [-4, 6, -3, 2, 1, 1, 7]
            .into_iter()
            .collect::<SegTree<SumQuery, _>>();

        assert_eq!(segtree.query(..), 10);
        assert_eq!(segtree.query(3..), 11);
        assert_eq!(segtree.query(3..6), 4);
        assert_eq!(segtree.query(..3), -1);

        segtree.update(2, 3);
        assert_eq!(segtree.query(..), 16);
        assert_eq!(segtree.query(3..), 11);
        assert_eq!(segtree.query(3..6), 4);
        assert_eq!(segtree.query(..3), 5);
        println!("{segtree:?}");
    }

    #[test]
    fn fill_test() {
        let mut segtree = [100, 200, 15, 40]
            .into_iter()
            .collect::<SegTree<MinQuery, _>>();

        assert_eq!(segtree.query(..), 15);
        segtree.fill(10);
        assert_eq!(segtree.query(..), 10);
        assert_eq!(segtree.query(1..), 10);
        assert_eq!(segtree.query(1..3), 10);

        segtree.fill_with(|| -20);
        assert_eq!(segtree.query(..), -20);
        assert_eq!(segtree.query(1..), -20);
        assert_eq!(segtree.query(1..3), -20);
    }

    #[test]
    fn upper_bound_test() {
        let mut segtree = [4, 2, 3, 1, 5, 6, 7]
            .into_iter()
            .collect::<SegTree<MinQuery, _>>();

        assert_eq!(segtree.upper_bound(&0), segtree.len());
        assert_eq!(segtree.upper_bound(&1), 3);
        assert_eq!(segtree.upper_bound(&2), 1);
        assert_eq!(segtree.upper_bound(&3), 1);
        assert_eq!(segtree.upper_bound(&4), 0);
        assert_eq!(segtree.upper_bound(&5), 0);

        segtree.update(1, 5);
        assert_eq!(segtree.upper_bound(&1), 3);
        assert_eq!(segtree.upper_bound(&2), 3);
        assert_eq!(segtree.upper_bound(&3), 2);
        assert_eq!(segtree.upper_bound(&4), 0);

        segtree.update(0, 5);
        assert_eq!(segtree.upper_bound(&4), 2);
        assert_eq!(segtree.upper_bound(&5), 0);

        assert_eq!(segtree.partition_point(7, |v| *v > 5), 8);
        segtree.update(7, 3);
        assert_eq!(segtree.partition_point(7, |v| *v > 5), 7);
    }

    #[test]
    fn max_query_test() {
        let mut segtree = [23i32, 12, -3, 0, 3, -2, 7, 8]
            .into_iter()
            .collect::<SegTree<MaxQuery, _>>();

        assert_eq!(segtree.query(..), 23);
        assert_eq!(segtree.query(1..), 12);
        assert_eq!(segtree.query(2..), 8);
        assert_eq!(segtree.query(1..6), 12);
        assert_eq!(segtree.query(2..6), 3);
        assert_eq!(segtree.query(2..=6), 7);

        segtree.update(2, 5);
        assert_eq!(segtree.query(..), 23);
        assert_eq!(segtree.query(2..), 8);
        assert_eq!(segtree.query(2..6), 5);
        assert_eq!(segtree.query(2..=6), 7);

        segtree.update(0, 10);
        assert_eq!(segtree.partition_point(0, |v| *v < 12), 1);
        assert_eq!(segtree.partition_point(0, |v| *v < 13), 8);
        assert_eq!(segtree.partition_point(1, |v| *v < 12), 1);
        assert_eq!(segtree.partition_point(2, |v| *v < 12), 8);
        assert_eq!(segtree.partition_point(2, |v| *v < 7), 6);
    }

    #[test]
    fn partition_point_test() {
        let segtree = [3u32, 5, 2, 1, 9, 11, 15, 3]
            .into_iter()
            .collect::<SegTree<SumQuery, _>>();

        assert_eq!(segtree.partition_point(0, |v| *v <= 20), 5);
        assert_eq!(segtree.partition_point(1, |v| *v <= 20), 5);
        assert_eq!(segtree.partition_point(4, |v| *v <= 25), 6);
        assert_eq!(segtree.partition_point(3, |v| *v <= 100), 8);
        assert_eq!(segtree.partition_point(8, |v| *v <= 20), 8);
    }

    #[test]
    #[should_panic]
    fn partition_point_panic() {
        let segtree = [3u32, 5, 2, 1, 9, 11, 15, 3]
            .into_iter()
            .collect::<SegTree<SumQuery, _>>();
        segtree.partition_point(9, |v| *v <= 20);
    }

    #[test]
    fn polynomial_query_test() {
        let segtree = SegTree::from_iter_query(
            PolynomialQuery::<_, SumQuery, ProdQuery>::new(7, 8),
            [1, 2, 3, 4, 5, 6, 7, 8],
        );
        assert_eq!(
            &segtree.tree[1..],
            &[7526268, 1534, 3134, 15, 31, 47, 63, 1, 2, 3, 4, 5, 6, 7, 8]
        );

        assert_eq!(segtree.query(1..4), 2 + 3 * 7 + 4 * 7 * 7);
        assert_eq!(segtree.query(1..5), 2 + 3 * 7 + 4 * 7 * 7 + 5 * 7 * 7 * 7);
        assert_eq!(segtree.query(3..6), 4 + 5 * 7 + 6 * 7 * 7);

        fn polynomial_slow(x: u64, c: &[u64], m: u64) -> u64 {
            let mut e = 1;
            c.iter()
                .map(|&c| {
                    let tmp = e * x % m;
                    c * std::mem::replace(&mut e, tmp) % m
                })
                .sum::<u64>() % m
        }

        let mut rng = StdRng::seed_from_u64(3940);
        let data = (0..1000)
            .map(|_| rng.gen_range(0..=1_000_000_000u64))
            .collect::<Vec<_>>();
        let x = rng.gen_range(1..1_000_000_000u64);
        let m = rng.gen_range(100_000_000u64..1_000_000_000);
        let segtree = SegTree::from_iter_query(
            PolynomialQuery::with_query(
                x,
                data.len().next_power_of_two(),
                Mod::new(SumQuery, m),
                Mod::new(ProdQuery, m),
            ),
            data.iter().copied(),
        );
        let range_iter = (0..200)
            .map(|_| {
                let l = rng.gen_range(0..data.len());
                let r = rng.gen_range(l + 1..=data.len());
                [l, r]
            });
        for [l, r] in range_iter {
            assert_eq!(
                segtree.query(l..r),
                polynomial_slow(
                    x,
                    &data[l..r],
                    m,
                )
            );
        }
    }
}
