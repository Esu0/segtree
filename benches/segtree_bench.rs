use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use rand::{rngs::StdRng, Rng, SeedableRng};
use segtree::{MinQuery, SegTree};

fn bench_segtree(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(100);
    let n = 100000;
    let mut get_segtree = || {
        std::iter::repeat_with(|| MinQuery(rng.gen_range(-1_000_000_000i64..1_000_000_000)))
            .take(n)
            .collect::<SegTree<_>>()
    };
    c.bench_function("segtree-query", |b| {
        let mut seg = get_segtree();
        black_box(&mut seg);
        b.iter(|| {
            black_box(seg.query(black_box(20..50255)));
        })
    });

    c.bench_function("segtree-update", |b| {
        let mut seg = get_segtree();
        black_box(&mut seg);
        b.iter(|| {
            seg.update(black_box(1023), black_box(MinQuery(30)));
        })
    });

    c.bench_function("segtree-query-update", |b| {
        let mut seg = get_segtree();
        black_box(&mut seg);
        b.iter(|| {
            black_box(seg.query(black_box(20..50255)));
            seg.update(black_box(1023), black_box(MinQuery(30)));
        })
    });

    c.bench_function("segtree-query-from0", |b| {
        let mut seg = get_segtree();
        black_box(&mut seg);
        b.iter(|| {
            black_box(seg.query(0..black_box(50255)));
        })
    });

    c.bench_function("segtree-partition-point", |b| {
        let mut seg = get_segtree();
        black_box(&mut seg);
        b.iter(|| {
            black_box(seg.partition_point(0, |x| x.0 > black_box(500_000_000)));
        })
    });

    c.bench_function("segtree-partition-point-range", |b| {
        let mut seg = get_segtree();
        black_box(&mut seg);
        b.iter(|| {
            black_box(seg.partition_point(black_box(4000), |x| x.0 > black_box(500_000_000)));
        })
    });
}

criterion_group!(benches, bench_segtree);
criterion_main!(benches);
