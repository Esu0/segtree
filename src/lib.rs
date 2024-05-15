use std::{
    alloc::Layout,
    borrow::Borrow,
    cmp::{Ordering, Reverse},
    mem::MaybeUninit,
    ops::{Add, Bound, Mul, RangeBounds},
    slice,
};

#[derive(Clone, Debug)]
pub struct SegTree<T> {
    tree: Box<[T]>,
}

impl<T: Query> SegTree<T> {
    /// half_len != 0
    unsafe fn make_tree_ptr(half_len: usize, f: impl FnOnce(*mut T) -> usize) -> *mut [T] {
        let len = half_len * 2;
        let ptr = std::alloc::alloc(Layout::array::<T>(len).unwrap()) as *mut T;
        {
            let data_ptr = ptr.add(half_len);
            let orig_len = f(data_ptr);
            for i in orig_len..half_len {
                data_ptr.add(i).write(T::IDENT);
            }
        }
        ptr.write(T::IDENT);
        Self::eval(ptr, half_len);

        std::ptr::slice_from_raw_parts_mut(ptr, len)
    }

    unsafe fn from_write_fn(half_len: usize, f: impl FnOnce(*mut T) -> usize) -> Self {
        Self {
            tree: Box::from_raw(Self::make_tree_ptr(half_len, f)),
        }
    }

    /// データのスライスからセグメント木を構築する。
    pub fn new(data: &[T]) -> Self {
        let orig_len = data.len();
        if orig_len == 0 {
            return Self { tree: Box::new([]) };
        }
        let half_len = orig_len.next_power_of_two();
        unsafe {
            Self::from_write_fn(half_len, |data_ptr| {
                for (i, data_i) in data.iter().enumerate() {
                    data_ptr.add(i).write(data_i.query(&T::IDENT))
                }
                orig_len
            })
        }
    }

    pub fn len(&self) -> usize {
        self.tree.len() / 2
    }

    pub fn is_empty(&self) -> bool {
        self.tree.is_empty()
    }

    unsafe fn eval(ptr: *mut T, half_len: usize) {
        for i in (1..half_len).rev() {
            ptr.add(i)
                .write((*ptr.add(i * 2)).query(&*ptr.add(i * 2 + 1)));
        }
    }

    /// 戻り値を`(l, r)`とすると以下が保証される。
    /// 
    /// * `l <= r <= self.len()`
    fn get_lr(&self, range: impl RangeBounds<usize>) -> (usize, usize) {
        let size = self.len();
        let l = match range.start_bound() {
            Bound::Excluded(s) => s.checked_add(1).unwrap_or_else(|| panic!("attempted to index slice from after maximum usize")),
            Bound::Included(s) => *s,
            Bound::Unbounded => 0,
        };
        let r = match range.end_bound() {
            Bound::Excluded(e) => *e,
            Bound::Included(e) => e.checked_add(1).unwrap_or_else(|| panic!("attempted to index slice up to maximum usize")),
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
            return T::IDENT;
        }
        l += self.len();
        r += self.len();
        let mut l_query = T::IDENT;
        let mut r_query = T::IDENT;
        while r - l > 2 {
            if l & 1 == 1 {
                l_query = l_query.query(&self.tree[l]);
                l += 1;
            }
            if r & 1 == 1 {
                r -= 1;
                r_query = self.tree[r].query(&r_query);
            }
            l >>= 1;
            r >>= 1;
        }
        if r - l == 2 {
            l_query
                .query(&self.tree[l])
                .query(&self.tree[l + 1])
                .query(&r_query)
        } else {
            l_query.query(&self.tree[l]).query(&r_query)
        }
    }

    /// 指定位置の要素をO(log(n))で更新する。
    pub fn update(&mut self, i: usize, val: T) {
        let mut i = i.checked_add(self.len()).unwrap_or_else(|| panic!("attempt to index slice maximum usize"));
        self.tree[i] = val;
        while i > 1 {
            i >>= 1;
            self.tree[i] = self.tree[i * 2].query(&self.tree[i * 2 + 1]);
        }
    }

    /// `pred(self.query(l..j))`が`true`となる最大の`j`をO(log(n))で求める。
    pub fn partition_point<P>(&self, l: usize, mut pred: P) -> usize
    where
        P: FnMut(&T) -> bool,
    {
        match l.cmp(&self.len()) {
            Ordering::Equal => return l,
            Ordering::Greater => panic!("index {l} out of range for slice of length {}", self.len()),
            _ => {}
        }
        let mut l = l.checked_add(self.len()).unwrap_or_else(|| panic!("attempt to index maximum usize"));
        let mut l_query = T::IDENT;
        loop {
            if l & 1 == 1 {
                let next_query = l_query.query(&self.tree[l]);
                let next_l = l + 1;
                if pred(&next_query) {
                    if next_l.is_power_of_two() {
                        return self.len();
                    } else {
                        l_query = next_query;
                    }
                } else {
                    break;
                }
                l = next_l;
            }
            l >>= 1;
        }
        loop {
            let next_l = l << 1;
            let Some(val) = self.tree.get(next_l) else {
                return l - self.len();
            };
            l = next_l;
            let next_query = l_query.query(val);
            if pred(&next_query) {
                l_query = next_query;
                l += 1;
            }
        }
    }
}

impl<T> SegTree<MinQuery<T>>
where
    MinQuery<T>: Query,
{
    // セグメント木の全要素を'val'で埋める。
    pub fn fill(&mut self, val: T)
    where
        T: Clone,
    {
        self.tree.fill(MinQuery(val));
    }

    /// セグメント木の全要素を`f`の戻り値で埋める。
    ///
    /// # Note
    /// 内部可変性を用いると`f`の戻り値が変化するようにすることもできるが、その場合はセグメント木は意味のない値で埋められ、
    /// クエリの結果も意味のない値になる。
    pub fn fill_with(&mut self, f: impl Fn() -> T) {
        self.tree.fill_with(|| MinQuery(f()))
    }

    /// `k`と等しいかそれより小さい要素の最左位置をO(log(n))で求める。
    ///
    /// self.query(..j) > kとなる最大のjを返す。
    pub fn upper_bound<Q: ?Sized>(&self, k: &Q) -> usize
    where
        T: Ord + Borrow<Q>,
        Q: Ord,
    {
        self.partition_point(0, |v| v.0.borrow() > k)
    }

    // fn upper_bound_rec<Q: ?Sized>(&self, k: &Q, l: usize, r: usize, i: usize) -> usize
    // where
    //     T: Ord + Borrow<Q>,
    //     Q: Ord,
    // {
    //     if self.tree[i].0.borrow() > k {
    //         // このノードの子はすべてkより大きいため、探索範囲外
    //         r
    //     } else if r - l == 1 {
    //         // k以下の値をもつ葉ノードであり、ここが最左位置となる
    //         l
    //     } else {
    //         // このノードの子にk以下の要素が含まれるため、再帰的に探索
    //         let mid = (l + r) / 2;
    //         let ch1 = 2 * i + 1;
    //         let ch2 = ch1 + 1;
    //         let result = self.upper_bound_rec(k, l, mid, ch1);
    //         if result == mid {
    //             self.upper_bound_rec(k, mid, r, ch2)
    //         } else {
    //             result
    //         }
    //     }
    // }
}

impl<I: Query> FromIterator<I> for SegTree<I> {
    fn from_iter<T: IntoIterator<Item = I>>(iter: T) -> Self {
        let iter = iter.into_iter();
        let (size_min, size_max) = iter.size_hint();
        if size_max == Some(0) {
            Self { tree: Box::new([]) }
        } else {
            assert_ne!(size_min, 0);
            let half_len_min = size_min.next_power_of_two();
            let half_len_max = size_max.map(usize::next_power_of_two);
            if Some(half_len_min) == half_len_max {
                let half_len = half_len_min;
                unsafe {
                    Self::from_write_fn(half_len, move |data_ptr| {
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
                    Self::from_write_fn(orig_len.next_power_of_two(), move |data_ptr| {
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
                    })
                }
            }
        }
    }
}

trait HasAddIdent {
    const IDENT: Self;
}

trait HasMulIdent {
    const IDENT: Self;
}

macro_rules! has_add_ident_num_impl {
    ($t:ty) => {
        impl HasAddIdent for $t {
            const IDENT: Self = 0;
        }
    };
}

macro_rules! has_mul_ident_num_impl {
    ($t:ty) => {
        impl HasMulIdent for $t {
            const IDENT: Self = 1;
        }
    }
}

has_add_ident_num_impl! {u8}
has_add_ident_num_impl! {u16}
has_add_ident_num_impl! {u32}
has_add_ident_num_impl! {u64}
has_add_ident_num_impl! {u128}
has_add_ident_num_impl! {i8}
has_add_ident_num_impl! {i16}
has_add_ident_num_impl! {i32}
has_add_ident_num_impl! {i64}
has_add_ident_num_impl! {i128}

has_mul_ident_num_impl! {u8}
has_mul_ident_num_impl! {u16}
has_mul_ident_num_impl! {u32}
has_mul_ident_num_impl! {u64}
has_mul_ident_num_impl! {u128}
has_mul_ident_num_impl! {i8}
has_mul_ident_num_impl! {i16}
has_mul_ident_num_impl! {i32}
has_mul_ident_num_impl! {i64}
has_mul_ident_num_impl! {i128}

trait HasMin {
    const MIN: Self;
}

macro_rules! has_min_num_impl {
    ($t:ty) => {
        impl HasMin for $t {
            const MIN: Self = <$t>::MIN;
        }
    };
}

has_min_num_impl! {u8}
has_min_num_impl! {u16}
has_min_num_impl! {u32}
has_min_num_impl! {u64}
has_min_num_impl! {u128}
has_min_num_impl! {i8}
has_min_num_impl! {i16}
has_min_num_impl! {i32}
has_min_num_impl! {i64}
has_min_num_impl! {i128}

trait HasMax {
    const MAX: Self;
}

macro_rules! has_max_num_impl {
    ($t:ty) => {
        impl HasMax for $t {
            const MAX: Self = <$t>::MAX;
        }
    };
}

has_max_num_impl! {u8}
has_max_num_impl! {u16}
has_max_num_impl! {u32}
has_max_num_impl! {u64}
has_max_num_impl! {u128}
has_max_num_impl! {i8}
has_max_num_impl! {i16}
has_max_num_impl! {i32}
has_max_num_impl! {i64}
has_max_num_impl! {i128}

impl<T: HasMax> HasMin for Reverse<T> {
    const MIN: Self = Self(<T as HasMax>::MAX);
}

impl<T: HasMin> HasMax for Reverse<T> {
    const MAX: Self = Self(<T as HasMin>::MIN);
}

pub trait Query {
    const IDENT: Self;
    fn query(&self, other: &Self) -> Self;
}

#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SumQuery<T>(pub T);

impl<T> SumQuery<T> {
    pub fn slice_from(slice: &[T]) -> &[Self] {
        let data = slice.as_ptr();
        let len = slice.len();
        unsafe { slice::from_raw_parts(data as _, len) }
    }
}

impl<T: Add<Output = T> + Clone + HasAddIdent> Query for SumQuery<T> {
    const IDENT: Self = Self(<T as HasAddIdent>::IDENT);
    fn query(&self, other: &Self) -> Self {
        Self(self.0.clone() + other.0.clone())
    }
}

#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ProdQuery<T>(pub T);

impl<T> ProdQuery<T> {
    pub fn slice_from(slice: &[T]) -> &[Self] {
        let data = slice.as_ptr();
        let len = slice.len();
        unsafe { slice::from_raw_parts(data as _, len) }
    }
}

impl<T: Mul<Output = T> + Clone + HasMulIdent> Query for ProdQuery<T> {
    const IDENT: Self = Self(<T as HasMulIdent>::IDENT);
    fn query(&self, other: &Self) -> Self {
        Self(self.0.clone() * other.0.clone())
    }
}

// #[repr(transparent)]
// #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
// pub struct MaxQuery<T>(pub T);

// impl<T> MaxQuery<T> {
//     pub fn slice_from(slice: &[T]) -> &[Self] {
//         let data = slice.as_ptr();
//         let len = slice.len();
//         unsafe { slice::from_raw_parts(data as _, len) }
//     }
// }

// impl<T: Ord + Clone + HasMin> Query for MaxQuery<T> {
//     const IDENT: Self = Self(<T as HasMin>::MIN);
//     fn query(&self, other: &Self) -> Self {
//         Self(std::cmp::max(self.0.clone(), other.0.clone()))
//     }
// }

#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MinQuery<T>(pub T);

impl<T> MinQuery<T> {
    pub fn slice_from(slice: &[T]) -> &[Self] {
        let data = slice.as_ptr();
        let len = slice.len();
        unsafe { slice::from_raw_parts(data as _, len) }
    }
}

impl<T: Ord + Clone + HasMax> Query for MinQuery<T> {
    const IDENT: Self = Self(<T as HasMax>::MAX);
    fn query(&self, other: &Self) -> Self {
        Self(std::cmp::min(self.0.clone(), other.0.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_test() {
        let segtree = SegTree::new(MinQuery::slice_from(&[1u32, 2, 3, 4, 5, 6]));
        println!("{segtree:#?}");
    }

    #[test]
    fn from_iter_test() {
        let segtree = [100, 200, 15, 40]
            .into_iter()
            .map(SumQuery)
            .collect::<SegTree<_>>();
        println!("{segtree:#?}");
    }

    #[test]
    fn sum_query_test() {
        let segtree = [-4, 6, -3, 2, 1, 1, 7]
            .into_iter()
            .map(SumQuery)
            .collect::<SegTree<_>>();

        assert_eq!(segtree.query(..).0, 10);
        assert_eq!(segtree.query(3..).0, 11);
        assert_eq!(segtree.query(3..6).0, 4);
        assert_eq!(segtree.query(..3).0, -1);

        assert_eq!(segtree.query(0..1).0, -4);
        assert_eq!(segtree.query(0..=0).0, -4);
        assert_eq!(segtree.query(0..=1).0, 2);
        assert_eq!(segtree.query(0..0).0, 0);
        assert_eq!(segtree.query(1..1).0, 0);
        assert_eq!(segtree.query(7..7).0, 0);
        assert_eq!(segtree.query(6..8).0, 7);
    }

    #[test]
    fn min_query_test() {
        let segtree = [23i32, 12, -3, 0, 3, -2, 7, 8]
            .into_iter()
            .map(MinQuery)
            .collect::<SegTree<_>>();

        assert_eq!(segtree.query(..).0, -3);
        assert_eq!(segtree.query(3..).0, -2);
        assert_eq!(segtree.query(..2).0, 12);
        assert_eq!(segtree.query(3..5).0, 0);

        assert_eq!(segtree.query(0..1).0, 23);
        assert_eq!(segtree.query(0..=0).0, 23);
        assert_eq!(segtree.query(0..=1).0, 12);
        assert_eq!(segtree.query(0..0).0, i32::MAX);
        assert_eq!(segtree.query(1..1).0, i32::MAX);
        assert_eq!(segtree.query(7..7).0, i32::MAX);
        assert_eq!(segtree.query(7..8).0, 8);
    }

    #[test]
    #[should_panic]
    fn out_of_bounds_test1() {
        let segtree = [1, 2, 3, 4, 5, 6, 7]
            .into_iter()
            .map(SumQuery)
            .collect::<SegTree<_>>();
        segtree.query(0..9);
    }

    #[test]
    #[should_panic]
    fn out_of_bounds_test2() {
        let segtree = [1, 2, 3, 4, 5, 6, 7]
            .into_iter()
            .map(SumQuery)
            .collect::<SegTree<_>>();
        segtree.query(9..);
    }

    #[test]
    #[should_panic]
    fn out_of_bounds_test3() {
        let segtree = [1, 2, 3, 4, 5, 6, 7]
            .into_iter()
            .map(SumQuery)
            .collect::<SegTree<_>>();
        segtree.query(0..=8);
    }

    #[test]
    #[should_panic]
    #[allow(clippy::reversed_empty_ranges)]
    fn out_of_bounds_test4() {
        let segtree = [1, 2, 3, 4, 5, 6, 7]
            .into_iter()
            .map(SumQuery)
            .collect::<SegTree<_>>();
        segtree.query(5..4);
    }

    #[test]
    fn update_test() {
        let mut segtree = [-4, 6, -3, 2, 1, 1, 7]
            .into_iter()
            .map(SumQuery)
            .collect::<SegTree<_>>();

        assert_eq!(segtree.query(..).0, 10);
        assert_eq!(segtree.query(3..).0, 11);
        assert_eq!(segtree.query(3..6).0, 4);
        assert_eq!(segtree.query(..3).0, -1);

        segtree.update(2, SumQuery(3));
        assert_eq!(segtree.query(..).0, 16);
        assert_eq!(segtree.query(3..).0, 11);
        assert_eq!(segtree.query(3..6).0, 4);
        assert_eq!(segtree.query(..3).0, 5);
        println!("{segtree:?}");
    }

    #[test]
    fn fill_test() {
        let mut segtree = [100, 200, 15, 40]
            .into_iter()
            .map(MinQuery)
            .collect::<SegTree<_>>();

        assert_eq!(segtree.query(..).0, 15);
        segtree.fill(10);
        assert_eq!(segtree.query(..).0, 10);
        assert_eq!(segtree.query(1..).0, 10);
        assert_eq!(segtree.query(1..3).0, 10);

        segtree.fill_with(|| -20);
        assert_eq!(segtree.query(..).0, -20);
        assert_eq!(segtree.query(1..).0, -20);
        assert_eq!(segtree.query(1..3).0, -20);
    }

    #[test]
    fn upper_bound_test() {
        let mut segtree = [4, 2, 3, 1, 5, 6, 7]
            .into_iter()
            .map(MinQuery)
            .collect::<SegTree<_>>();

        assert_eq!(segtree.upper_bound(&0), segtree.len());
        assert_eq!(segtree.upper_bound(&1), 3);
        assert_eq!(segtree.upper_bound(&2), 1);
        assert_eq!(segtree.upper_bound(&3), 1);
        assert_eq!(segtree.upper_bound(&4), 0);
        assert_eq!(segtree.upper_bound(&5), 0);

        segtree.update(1, MinQuery(5));
        assert_eq!(segtree.upper_bound(&1), 3);
        assert_eq!(segtree.upper_bound(&2), 3);
        assert_eq!(segtree.upper_bound(&3), 2);
        assert_eq!(segtree.upper_bound(&4), 0);

        segtree.update(0, MinQuery(5));
        assert_eq!(segtree.upper_bound(&4), 2);
        assert_eq!(segtree.upper_bound(&5), 0);

        assert_eq!(segtree.partition_point(7, |v| v.0 > 5), 8);
        segtree.update(7, MinQuery(3));
        assert_eq!(segtree.partition_point(7, |v| v.0 > 5), 7);
    }

    #[test]
    fn partition_point_test() {
        let segtree = [3u32, 5, 2, 1, 9, 11, 15, 3].into_iter().map(SumQuery).collect::<SegTree<_>>();

        assert_eq!(segtree.partition_point(0, |v| v.0 <= 20), 5);
        assert_eq!(segtree.partition_point(1, |v| v.0 <= 20), 5);
        assert_eq!(segtree.partition_point(4, |v| v.0 <= 25), 6);
        assert_eq!(segtree.partition_point(3, |v| v.0 <= 100), 8);
        assert_eq!(segtree.partition_point(8, |v| v.0 <= 20), 8);
    }

    #[test]
    #[should_panic]
    fn partition_point_panic() {
        let segtree = [3u32, 5, 2, 1, 9, 11, 15, 3].into_iter().map(SumQuery).collect::<SegTree<_>>();
        segtree.partition_point(9, |v| v.0 <= 20);
    }
}
