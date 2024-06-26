use std::{cell::RefCell, hint::black_box, iter};

use criterion::{criterion_group, criterion_main, Criterion};
use rand::{
    distributions::uniform::{SampleRange, SampleUniform},
    rngs::StdRng,
    Rng, SeedableRng,
};
use segtree::{
    query::{MinQuery, Mod, PolynomialQuery, ProdQuery, QueryWith, SumQuery},
    SegTree,
};

/// ランダムなセグメント木を生成する
fn get_segtree<Q: QueryWith<i64>>(n: usize, query: Q) -> SegTree<Q, i64> {
    thread_local! {
        static RNG: RefCell<StdRng> = RefCell::new(StdRng::seed_from_u64(100));
    }
    RNG.with(|rng| {
        let mut rng = rng.borrow_mut();
        SegTree::from_iter_query(
            query,
            black_box(
                iter::repeat_with(|| rng.gen_range(-1_000_000_000i64..=1_000_000_000)).take(n),
            ),
        )
    })
}

fn get_segtree_range<T: SampleUniform, Q: QueryWith<T>>(
    n: usize,
    query: Q,
    range: impl SampleRange<T> + Clone,
) -> SegTree<Q, T> {
    thread_local! {
        static RNG: RefCell<StdRng> = RefCell::new(StdRng::seed_from_u64(488291));
    }
    RNG.with(|rng| {
        let mut rng = rng.borrow_mut();
        SegTree::from_iter_query(
            query,
            black_box(iter::repeat_with(|| rng.gen_range(range.clone())).take(n)),
        )
    })
}
/// ランダムな`Range<usize>`を生成する
fn get_random_range(n: usize, m: usize) -> Vec<[usize; 2]> {
    thread_local! {
        static RNG: RefCell<StdRng> = RefCell::new(StdRng::seed_from_u64(100));
    }
    RNG.with(|rng| {
        let mut rng = rng.borrow_mut();
        iter::repeat_with(|| {
            let l = rng.gen_range(0..n);
            [l, rng.gen_range(l + 1..=n)]
        })
        .take(m)
        .collect::<Vec<_>>()
    })
}

/// ランダムなインデックスを生成する
fn get_random_index(n: usize, m: usize) -> Vec<usize> {
    thread_local! {
        static RNG: RefCell<StdRng> = RefCell::new(StdRng::seed_from_u64(2399));
    }
    RNG.with(|rng| {
        let mut rng = rng.borrow_mut();
        iter::repeat_with(|| rng.gen_range(0..n))
            .take(m)
            .collect::<Vec<_>>()
    })
}

/// ランダムなデータを生成する
fn get_random_data(n: usize) -> Vec<i64> {
    thread_local! {
        static RNG: RefCell<StdRng> = RefCell::new(StdRng::seed_from_u64(103));
    }
    RNG.with(|rng| {
        let mut rng = rng.borrow_mut();
        iter::repeat_with(|| rng.gen_range(-1_000_000_000i64..1_000_000_000))
            .take(n)
            .collect::<Vec<_>>()
    })
}

fn random_num<T: SampleUniform>(range: impl SampleRange<T>) -> T {
    thread_local! {
        static RNG: RefCell<StdRng> = RefCell::new(StdRng::seed_from_u64(202));
    }
    RNG.with(|rng| {
        let mut rng = rng.borrow_mut();
        rng.gen_range(range)
    })
}

fn bench_segtree(c: &mut Criterion) {
    let n = 100000;

    c.bench_function("segtree-query", |b| {
        let seg = get_segtree(n, MinQuery);
        let ranges = get_random_range(n, 200000);
        let mut range_iter = ranges.iter().copied().cycle();
        b.iter(|| {
            let [l, r] = range_iter.next().unwrap();
            black_box(seg.query(black_box(l..r)));
        })
    });

    c.bench_function("segtree-update", |b| {
        let mut seg = get_segtree(n, MinQuery);
        let indice = get_random_index(n, 200000);
        let data = get_random_data(200000);
        let mut index_iter = indice.iter().copied().cycle();
        let mut data_iter = data.iter().copied().cycle();
        b.iter(|| {
            seg.update(
                black_box(index_iter.next().unwrap()),
                black_box(data_iter.next().unwrap()),
            );
        })
    });

    c.bench_function("segtree-query-update", |b| {
        let mut seg = get_segtree(n, MinQuery);
        let ranges = get_random_range(n, 200000);
        let indice = get_random_index(n, 200000);
        let data = get_random_data(200000);
        let mut range_iter = ranges.iter().copied().cycle();
        let mut index_iter = indice.iter().copied().cycle();
        let mut data_iter = data.iter().copied().cycle();

        b.iter(|| {
            let [l, r] = black_box(range_iter.next().unwrap());
            let i = black_box(index_iter.next().unwrap());
            let d = black_box(data_iter.next().unwrap());

            black_box(seg.query(l..r));
            seg.update(i, d);
        })
    });

    c.bench_function("segtree-query-from0", |b| {
        let seg = get_segtree(n, MinQuery);
        let indice = get_random_index(n, 200000);
        let mut index_iter = indice.iter().copied().cycle();
        b.iter(|| {
            let i = black_box(index_iter.next().unwrap());
            black_box(seg.query(0..i));
        })
    });

    c.bench_function("segtree-partition-point", |b| {
        let seg = get_segtree(n, MinQuery);
        let data = get_random_data(n);
        let mut data_iter = data.iter().cycle();
        b.iter(|| {
            let d = black_box(*data_iter.next().unwrap());
            black_box(seg.partition_point(0, |x| *x > d));
        })
    });

    c.bench_function("segtree-partition-point-range", |b| {
        let seg = get_segtree(n, MinQuery);
        let data = get_random_data(n);
        let indice = get_random_index(n, 200000);
        let mut data_iter = data.iter().copied().cycle();
        let mut index_iter = indice.iter().copied().cycle();
        b.iter(|| {
            let i = black_box(index_iter.next().unwrap());
            let d = black_box(data_iter.next().unwrap());
            black_box(seg.partition_point(i, |x| *x > d));
        })
    });

    c.bench_function("segtree-polynomial-query", |b| {
        let seg = get_segtree_range(
            n,
            PolynomialQuery::with_query(
                random_num(1u64..1_000_000_000),
                n,
                Mod::new(SumQuery, 1_000_000_007),
                Mod::new(ProdQuery, 1_000_000_007),
            ),
            0u64..=1_000_000_000,
        );
        let ranges = get_random_range(n, 200_000);
        let mut range_iter = ranges.iter().copied().cycle();
        b.iter(|| {
            let [l, r] = black_box(range_iter.next().unwrap());
            black_box(seg.query(l..r));
        })
    });
}

fn bench_ac_lib_segtree(c: &mut Criterion) {
    let n = 100_000;
    c.bench_function("ac-lib-segtree-query", |b| {
        let data = get_random_data(n);
        let seg = ac_library::Segtree::<ac_library::segtree::Min<_>>::from(data);
        let ranges = get_random_range(n, 200_000);
        let mut range_iter = ranges.iter().copied().cycle();
        b.iter(|| {
            let [l, r] = range_iter.next().unwrap();
            black_box(seg.prod(l..r));
        })
    });

    c.bench_function("ac-lib-segtree-query-from0", |b| {
        let data = get_random_data(n);
        let seg = ac_library::Segtree::<ac_library::segtree::Min<_>>::from(data);
        let indice = get_random_index(n, 200_000);
        let mut index_iter = indice.iter().copied().cycle();
        b.iter(|| {
            let i = index_iter.next().unwrap();
            black_box(seg.prod(0..i));
        })
    });

    c.bench_function("ac-lib-segtree-update", |b| {
        let data = get_random_data(n);
        let mut seg = ac_library::Segtree::<ac_library::segtree::Min<_>>::from(data);
        let indice = get_random_index(n, 200_000);
        let data = get_random_data(200_000);
        let mut index_iter = indice.iter().copied().cycle();
        let mut data_iter = data.iter().copied().cycle();
        b.iter(|| {
            seg.set(index_iter.next().unwrap(), data_iter.next().unwrap());
        })
    });

    c.bench_function("ac-lib-segtree-query-update", |b| {
        let mut seg = ac_library::Segtree::<ac_library::segtree::Min<_>>::from(get_random_data(n));
        let ranges = get_random_range(n, 200000);
        let indice = get_random_index(n, 200000);
        let data = get_random_data(200000);
        let mut range_iter = ranges.iter().copied().cycle();
        let mut index_iter = indice.iter().copied().cycle();
        let mut data_iter = data.iter().copied().cycle();

        b.iter(|| {
            let [l, r] = black_box(range_iter.next().unwrap());
            let i = black_box(index_iter.next().unwrap());
            let d = black_box(data_iter.next().unwrap());

            black_box(seg.prod(l..r));
            seg.set(i, d);
        })
    });
}
criterion_group!(benches, bench_segtree, bench_ac_lib_segtree);
criterion_main!(benches);
