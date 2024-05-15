use std::{cell::RefCell, hint::black_box, iter};

use criterion::{criterion_group, criterion_main, Criterion};
use rand::{rngs::StdRng, Rng, SeedableRng};
use segtree::{
    query::{MinQuery, Query},
    SegTree,
};

/// ランダムなセグメント木を生成する
fn get_segtree<Q: Query<i64>>(n: usize, query: Q) -> SegTree<Q, i64> {
    thread_local! {
        static RNG: RefCell<StdRng> = RefCell::new(StdRng::seed_from_u64(100));
    }
    RNG.with(|rng| {
        let mut rng = rng.borrow_mut();
        SegTree::from_iter_query(
            query,
            iter::repeat_with(|| rng.gen_range(-1_000_000_000i64..=1_000_000_000)).take(n),
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
        static RNG: RefCell<StdRng> = RefCell::new(StdRng::seed_from_u64(102));
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
        static RNG: RefCell<StdRng> = RefCell::new(StdRng::seed_from_u64(102));
    }
    RNG.with(|rng| {
        let mut rng = rng.borrow_mut();
        iter::repeat_with(|| rng.gen_range(-1_000_000_000i64..1_000_000_000))
            .take(n)
            .collect::<Vec<_>>()
    })
}

fn bench_segtree(c: &mut Criterion) {
    let n = 100000;

    c.bench_function("segtree-query", |b| {
        let mut seg = get_segtree(n, MinQuery);
        let ranges = get_random_range(n, 200000);
        black_box(&mut seg);
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
        black_box(&mut seg);
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
        black_box(&mut seg);
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
        let mut seg = get_segtree(n, MinQuery);
        let indice = get_random_index(n, 200000);
        black_box(&mut seg);
        let mut index_iter = indice.iter().copied().cycle();
        b.iter(|| {
            let i = black_box(index_iter.next().unwrap());
            black_box(seg.query(0..i));
        })
    });

    c.bench_function("segtree-partition-point", |b| {
        let mut seg = get_segtree(n, MinQuery);
        let data = get_random_data(n);
        black_box(&mut seg);
        let mut data_iter = data.iter().cycle();
        b.iter(|| {
            let d = black_box(*data_iter.next().unwrap());
            black_box(seg.partition_point(0, |x| *x > d));
        })
    });

    c.bench_function("segtree-partition-point-range", |b| {
        let mut seg = get_segtree(n, MinQuery);
        let data = get_random_data(n);
        let indice = get_random_index(n, 200000);
        black_box(&mut seg);
        let mut data_iter = data.iter().copied().cycle();
        let mut index_iter = indice.iter().copied().cycle();
        b.iter(|| {
            let i = black_box(index_iter.next().unwrap());
            let d = black_box(data_iter.next().unwrap());
            black_box(seg.partition_point(i, |x| *x > d));
        })
    });
}

criterion_group!(benches, bench_segtree);
criterion_main!(benches);
