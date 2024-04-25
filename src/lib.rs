use core::slice;
use std::{alloc::Layout, cmp::Reverse, mem::MaybeUninit, ops::Add};

#[derive(Clone, Debug)]
pub struct SegTree<T> {
    tree: Box<[T]>,
}

impl<T: Query> SegTree<T> {
    /// half_len != 0
    unsafe fn make_tree_ptr(half_len: usize, f: impl FnOnce(*mut T) -> usize) -> *mut [T] {
        let len = half_len * 2 - 1;
        let ptr = std::alloc::alloc(Layout::array::<T>(len).unwrap()) as *mut T;
        {
            let data_ptr = ptr.add(half_len - 1);
            let orig_len = f(data_ptr);
            for i in orig_len..half_len {
                data_ptr.add(i).write(T::IDENT);
            }
        }

        Self::eval(ptr, half_len);

        std::ptr::slice_from_raw_parts_mut(ptr, len)
    }

    unsafe fn from_write_fn(half_len: usize, f: impl FnOnce(*mut T) -> usize) -> Self {
        Self {tree:Box::from_raw(Self::make_tree_ptr(half_len, f))}
    }
    pub fn new(data: &[T]) -> Self {
        let orig_len = data.len();
        if orig_len == 0 {
            return Self {
                tree: Box::new([]),
            };
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

    unsafe fn eval(ptr: *mut T, half_len: usize) {
        for i in (0..(half_len - 1)).rev() {
            ptr.add(i).write((*ptr.add(i * 2 + 1)).query(&*ptr.add(i * 2 + 2)));
        }
    }
}

impl<I: Query> FromIterator<I> for SegTree<I> {
    fn from_iter<T: IntoIterator<Item = I>>(iter: T) -> Self {
        let iter = iter.into_iter();
        let (size_min, size_max) = iter.size_hint();
        if size_max == Some(0) {
            Self {
                tree: Box::new([]),
            }
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
                        drop(Vec::from_raw_parts(src as *mut MaybeUninit<I>, orig_len, cap));
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

macro_rules! has_ident_num_impl {
    ($t:ty) => {
        impl HasAddIdent for $t {
            const IDENT: Self = 0;
        }
    };
}

has_ident_num_impl!{u8}
has_ident_num_impl!{u16}
has_ident_num_impl!{u32}
has_ident_num_impl!{u64}
has_ident_num_impl!{u128}
has_ident_num_impl!{i8}
has_ident_num_impl!{i16}
has_ident_num_impl!{i32}
has_ident_num_impl!{i64}
has_ident_num_impl!{i128}

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

has_min_num_impl!{u8}
has_min_num_impl!{u16}
has_min_num_impl!{u32}
has_min_num_impl!{u64}
has_min_num_impl!{u128}
has_min_num_impl!{i8}
has_min_num_impl!{i16}
has_min_num_impl!{i32}
has_min_num_impl!{i64}
has_min_num_impl!{i128}

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

has_max_num_impl!{u8}
has_max_num_impl!{u16}
has_max_num_impl!{u32}
has_max_num_impl!{u64}
has_max_num_impl!{u128}
has_max_num_impl!{i8}
has_max_num_impl!{i16}
has_max_num_impl!{i32}
has_max_num_impl!{i64}
has_max_num_impl!{i128}

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
        unsafe {
            slice::from_raw_parts(data as _, len)
        }
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
pub struct MaxQuery<T>(pub T);

impl<T> MaxQuery<T> {
    pub fn slice_from(slice: &[T]) -> &[Self] {
        let data = slice.as_ptr();
        let len = slice.len();
        unsafe {
            slice::from_raw_parts(data as _, len)
        }
    }
}

impl<T: Ord + Clone + HasMin> Query for MaxQuery<T> {
    const IDENT: Self = Self(<T as HasMin>::MIN);
    fn query(&self, other: &Self) -> Self {
        Self(std::cmp::max(self.0.clone(), other.0.clone()))
    }
}

#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MinQuery<T>(pub T);

impl<T> MinQuery<T> {
    pub fn slice_from(slice: &[T]) -> &[Self] {
        let data = slice.as_ptr();
        let len = slice.len();
        unsafe {
            slice::from_raw_parts(data as _, len)
        }
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
        let segtree = [100, 200, 15, 40].into_iter().map(SumQuery).collect::<SegTree<_>>();
        println!("{segtree:#?}");
    }
}
