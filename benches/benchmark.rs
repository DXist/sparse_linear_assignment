use criterion::{black_box, criterion_group, criterion_main, Criterion};
use sslap::cumulative_idxs;

pub fn cumulative_idxs_benchmark(c: &mut Criterion) {
    c.bench_function("cum idx 7", |b| {
        b.iter(|| cumulative_idxs(black_box(&[0, 0, 0, 1, 1, 1, 1])))
    });
}

criterion_group!(benches, cumulative_idxs_benchmark);
criterion_main!(benches);
