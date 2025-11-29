use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rusttorch_core::{Tensor, DType};

fn bench_tensor_creation(c: &mut Criterion) {
    c.bench_function("tensor_zeros_100x100", |b| {
        b.iter(|| {
            black_box(Tensor::zeros(&[100, 100], DType::Float32))
        })
    });

    c.bench_function("tensor_ones_100x100", |b| {
        b.iter(|| {
            black_box(Tensor::ones(&[100, 100], DType::Float32))
        })
    });
}

fn bench_tensor_properties(c: &mut Criterion) {
    let tensor = Tensor::zeros(&[1000, 1000], DType::Float32);

    c.bench_function("tensor_shape", |b| {
        b.iter(|| {
            black_box(tensor.shape())
        })
    });

    c.bench_function("tensor_numel", |b| {
        b.iter(|| {
            black_box(tensor.numel())
        })
    });
}

criterion_group!(benches, bench_tensor_creation, bench_tensor_properties);
criterion_main!(benches);
