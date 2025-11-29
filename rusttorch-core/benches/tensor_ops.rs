use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use rusttorch_core::{Tensor, DType};
use rusttorch_core::ops::{
    elementwise::{add, mul, sub, div, add_scalar, mul_scalar},
    reduction::{sum, mean, max, min, sum_dim, mean_dim},
    activation::{relu, leaky_relu, sigmoid, tanh, gelu, softmax},
    matrix::{matmul, transpose, reshape},
};

fn bench_tensor_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_creation");

    for size in [100, 500, 1000].iter() {
        group.bench_with_input(BenchmarkId::new("zeros", size), size, |b, &s| {
            b.iter(|| {
                black_box(Tensor::zeros(&[s, s], DType::Float32))
            })
        });

        group.bench_with_input(BenchmarkId::new("ones", size), size, |b, &s| {
            b.iter(|| {
                black_box(Tensor::ones(&[s, s], DType::Float32))
            })
        });
    }

    group.finish();
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

fn bench_elementwise_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("elementwise_ops");

    for size in [100, 500, 1000].iter() {
        let a = Tensor::ones(&[*size, *size], DType::Float32);
        let b = Tensor::ones(&[*size, *size], DType::Float32);

        group.bench_with_input(BenchmarkId::new("add", size), size, |bench, _| {
            bench.iter(|| {
                black_box(add(&a, &b).unwrap())
            })
        });

        group.bench_with_input(BenchmarkId::new("mul", size), size, |bench, _| {
            bench.iter(|| {
                black_box(mul(&a, &b).unwrap())
            })
        });

        group.bench_with_input(BenchmarkId::new("sub", size), size, |bench, _| {
            bench.iter(|| {
                black_box(sub(&a, &b).unwrap())
            })
        });

        group.bench_with_input(BenchmarkId::new("div", size), size, |bench, _| {
            bench.iter(|| {
                black_box(div(&a, &b).unwrap())
            })
        });

        group.bench_with_input(BenchmarkId::new("add_scalar", size), size, |bench, _| {
            bench.iter(|| {
                black_box(add_scalar(&a, 2.0))
            })
        });

        group.bench_with_input(BenchmarkId::new("mul_scalar", size), size, |bench, _| {
            bench.iter(|| {
                black_box(mul_scalar(&a, 2.0))
            })
        });
    }

    group.finish();
}

fn bench_reduction_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("reduction_ops");

    for size in [100, 500, 1000].iter() {
        let tensor = Tensor::ones(&[*size, *size], DType::Float32);

        group.bench_with_input(BenchmarkId::new("sum", size), size, |bench, _| {
            bench.iter(|| {
                black_box(sum(&tensor))
            })
        });

        group.bench_with_input(BenchmarkId::new("mean", size), size, |bench, _| {
            bench.iter(|| {
                black_box(mean(&tensor))
            })
        });

        group.bench_with_input(BenchmarkId::new("max", size), size, |bench, _| {
            bench.iter(|| {
                black_box(max(&tensor))
            })
        });

        group.bench_with_input(BenchmarkId::new("min", size), size, |bench, _| {
            bench.iter(|| {
                black_box(min(&tensor))
            })
        });

        group.bench_with_input(BenchmarkId::new("sum_dim", size), size, |bench, _| {
            bench.iter(|| {
                black_box(sum_dim(&tensor, 0).unwrap())
            })
        });

        group.bench_with_input(BenchmarkId::new("mean_dim", size), size, |bench, _| {
            bench.iter(|| {
                black_box(mean_dim(&tensor, 0).unwrap())
            })
        });
    }

    group.finish();
}

fn bench_activation_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("activation_ops");

    for size in [100, 500, 1000].iter() {
        let tensor = Tensor::ones(&[*size, *size], DType::Float32);

        group.bench_with_input(BenchmarkId::new("relu", size), size, |bench, _| {
            bench.iter(|| {
                black_box(relu(&tensor))
            })
        });

        group.bench_with_input(BenchmarkId::new("leaky_relu", size), size, |bench, _| {
            bench.iter(|| {
                black_box(leaky_relu(&tensor, 0.01))
            })
        });

        group.bench_with_input(BenchmarkId::new("sigmoid", size), size, |bench, _| {
            bench.iter(|| {
                black_box(sigmoid(&tensor).unwrap())
            })
        });

        group.bench_with_input(BenchmarkId::new("tanh", size), size, |bench, _| {
            bench.iter(|| {
                black_box(tanh(&tensor).unwrap())
            })
        });

        group.bench_with_input(BenchmarkId::new("gelu", size), size, |bench, _| {
            bench.iter(|| {
                black_box(gelu(&tensor).unwrap())
            })
        });

        group.bench_with_input(BenchmarkId::new("softmax", size), size, |bench, _| {
            bench.iter(|| {
                black_box(softmax(&tensor, 1).unwrap())
            })
        });
    }

    group.finish();
}

fn bench_matrix_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_ops");

    // Test different matrix sizes
    let sizes = [(64, 64, 64), (128, 128, 128), (256, 256, 256)];

    for (m, k, n) in sizes.iter() {
        let a = Tensor::ones(&[*m, *k], DType::Float32);
        let b = Tensor::ones(&[*k, *n], DType::Float32);

        group.bench_with_input(
            BenchmarkId::new("matmul", format!("{}x{}x{}", m, k, n)),
            &(m, k, n),
            |bench, _| {
                bench.iter(|| {
                    black_box(matmul(&a, &b).unwrap())
                })
            },
        );
    }

    // Transpose benchmarks
    for size in [100, 500, 1000].iter() {
        let tensor = Tensor::ones(&[*size, *size], DType::Float32);

        group.bench_with_input(BenchmarkId::new("transpose", size), size, |bench, _| {
            bench.iter(|| {
                black_box(transpose(&tensor))
            })
        });
    }

    // Reshape benchmarks
    for size in [100, 500, 1000].iter() {
        let tensor = Tensor::ones(&[*size, *size], DType::Float32);
        let new_shape = [*size * *size / 2, 2];

        group.bench_with_input(BenchmarkId::new("reshape", size), size, |bench, _| {
            bench.iter(|| {
                black_box(reshape(&tensor, &new_shape).unwrap())
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_tensor_creation,
    bench_tensor_properties,
    bench_elementwise_ops,
    bench_reduction_ops,
    bench_activation_ops,
    bench_matrix_ops
);
criterion_main!(benches);
