use std::cmp::Reverse;

pub(crate) trait HasAddIdent {
    const IDENT: Self;
}

pub(crate) trait HasMulIdent {
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
    };
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

pub(super) trait HasMin {
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

pub(super) trait HasMax {
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
