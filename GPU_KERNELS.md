# GPU Kernels — Work Queue and Spec

This document is the spec for porting rusttorch's CPU kernels to CUDA. It exists on the `gpu-device-abstraction` branch ahead of any GPU code so that when A100/H100 access arrives, the kernel list, signatures, and priorities are already decided.

## Target hardware

- **Primary**: A100 (SM 80) and H100 (SM 90). f32 first, bf16/f16 second.
- **Secondary**: any Ampere+ consumer card (RTX 3090, 4090) for cheap iteration.
- **Out of scope**: Pascal/Turing (no TF32), Apple Metal, AMD ROCm. `wgpu` path not pursued — too much ML ecosystem churn for not enough return.

## Backend architecture

The `Backend` trait in `rusttorch-core/src/backend/mod.rs` is the dispatch surface. The CPU implementation (`NdArrayBackend`) exists today and passes every rusttorch test; the CUDA implementation (`CudaBackend`, unwritten) will be selected at runtime based on `Tensor::device()`.

```rust
pub trait Backend {
    fn device(&self) -> Device;
    fn matmul(&self, a: &Variable, b: &Variable) -> Result<Variable>;
    fn add(&self, a: &Variable, b: &Variable) -> Result<Variable>;
    fn mul(&self, a: &Variable, b: &Variable) -> Result<Variable>;
    fn relu(&self, x: &Variable) -> Variable;
    fn gelu(&self, x: &Variable) -> Variable;
    fn log_softmax(&self, x: &Variable, dim: usize) -> Result<Variable>;
    fn layer_norm(&self, x: &Variable, norm_size: usize,
                  weight: Option<&Variable>, bias: Option<&Variable>,
                  eps: f32) -> Result<Variable>;
    fn scalar(&self, value: f32) -> Tensor;
}
```

This method set intentionally covers the GPT-2 inference critical path. The full Backend trait will grow as kernels land.

## Dependency plan

- **cudarc** (Rust CUDA driver bindings) — minimal, no C++ compiler dependency, handles PTX loading cleanly. Alternative to `cust` (stale) and `candle` (tight HF coupling).
- Guarded behind the `cuda` cargo feature. CPU builds must not pay any dependency cost.
- Kernels as inline PTX or `.cu` files compiled by `nvcc` at build-time via `build.rs`. First pass: keep kernels in `.cu` files for readability; optimize with PTX intrinsics later.

## Kernel queue (ordered by GPT-2 dependency)

Every kernel below is needed to run GPT-2 inference end-to-end on GPU. Order is dependency-first (foundations before compositions) and complexity-second (simpler kernels first).

### Tier 1 — Must-have to match CPU parity on a forward pass

| # | Kernel | Input / Output shape | Notes |
|---|---|---|---|
| 1 | `matmul_f32` | `[M, K] x [K, N] -> [M, N]` | Start with cuBLAS `gemmEx`. Custom tile-based kernel only if profiling shows cuBLAS is suboptimal (it rarely is). |
| 2 | `add_elementwise_f32` | broadcasted `[*] + [*] -> [*]` | Trivial grid-stride loop. Will later fuse with bias-add. |
| 3 | `mul_elementwise_f32` | same | Trivial. Fuse with `add` for fused-multiply-add. |
| 4 | `relu_f32` | `[*] -> [*]` | Trivial. In-place variant as a follow-up. |
| 5 | `gelu_f32` | `[*] -> [*]` | Tanh approximation to match rusttorch's CPU impl. Numerical drift ≤1e-5. |
| 6 | `softmax_f32` | `[B, S]` along `dim=-1` | Numerically stable: subtract rowwise max before exp. One thread block per row. Use warp reductions. |
| 7 | `log_softmax_f32` | same | Required for cross-entropy loss and generation sampling. |
| 8 | `layer_norm_f32` | `[*, D]` over last `norm_size=D` | Fused mean/var + normalize + scale + bias in a single kernel. cub::BlockReduce for stats. |
| 9 | `embedding_lookup_f32` | `indices [B, S] + weight [V, D] -> [B, S, D]` | Gather. Trivial kernel, one thread per output element. |
| 10 | `copy_h2d_f32` / `copy_d2h_f32` | Host/device transfer | cudarc's `htod_copy_into` / `dtoh_sync_copy`. Not a kernel, but needed. |

### Tier 2 — Fused kernels for performance

| # | Kernel | Why |
|---|---|---|
| 11 | `fused_qkv_projection` | 3 separate matmuls → 1 matmul against stacked weight. Already done on CPU (Gpt2 model Conv1D path); GPU version should match. |
| 12 | `flash_attention_v2` | The whole `Q @ K^T * scale -> softmax -> @ V` pipeline fused with tiled softmax. Reference: Dao 2023 (Flash Attention 2). ~400 LOC. Massive speedup over naive attention. |
| 13 | `fused_ffn_block` | Linear → GELU → Linear fused. Saves 4x activation round-trips through HBM. |
| 14 | `fused_layernorm_add_residual` | `residual + layernorm(x)` in one pass. Cuts 1 HBM read. |

### Tier 3 — Beyond GPT-2 inference

| # | Kernel | Notes |
|---|---|---|
| 15 | `conv2d_im2col` | Uses matmul + reshape. Implement after matmul is stable. Needed for CIFAR-10 ResNet on GPU. |
| 16 | `batchnorm2d_train` | Fused mean/var/normalize/running-stats. |
| 17 | `max_pool2d` / `avg_pool2d` / `adaptive_avg_pool2d` | Standard pool kernels. |
| 18 | `dropout_mask` | Uses cuRAND Philox4x32. |
| 19 | `top_k_sampling` / `top_p_sampling` | For GPT-2 `generate()`. Use CUB radix sort or scan. |
| 20 | `adam_step` | Fused parameter update: `m ← β1 m + (1-β1) g`, `v ← β2 v + (1-β2) g²`, `p ← p - lr * m̂ / (√v̂ + ε)`. One kernel per param-group launch. |

## Correctness strategy

Every GPU kernel lands with a CPU-parity test:

```rust
#[cfg(all(test, feature = "cuda"))]
fn test_matmul_cpu_cuda_parity() {
    let cpu = NdArrayBackend::new();
    let cuda = CudaBackend::new(0).unwrap();
    let a = Variable::new(random_tensor([64, 128]), false);
    let b = Variable::new(random_tensor([128, 96]), false);
    let cpu_out = cpu.matmul(&a, &b).unwrap().tensor().to_vec_f32();
    let cuda_out = cuda.matmul(&a.to_device(Device::Cuda(0))?,
                               &b.to_device(Device::Cuda(0))?)?
                       .tensor().cpu().to_vec_f32();
    assert_allclose(&cpu_out, &cuda_out, atol=1e-4);
}
```

Tolerance is kernel-specific:
- matmul (with TF32 allowed on Ampere+): `atol=1e-3`, `rtol=1e-4`
- softmax / layer_norm: `atol=1e-5`
- gelu / elementwise: `atol=1e-6`

## Performance targets (first-cut, A100 40GB)

- **GPT-2 124M inference**: ≥200 tok/s at batch=1, seq=256. CPU baseline is 3.0 tok/s, so ≥60x.
- **matmul `[1024, 4096] @ [4096, 4096]`**: ≥80% of cuBLAS peak (~150 TFLOPS).
- **LayerNorm `[32, 512, 768]`**: memory-bound, ≥80% of HBM peak bandwidth.

Track in `benches/gpu_kernels.rs` once the CUDA backend exists.

## Build plumbing

1. `build.rs` detects CUDA via `$CUDA_HOME` or `nvcc --version`. If found and `cuda` feature is on, it compiles `.cu` files into PTX and emits `cargo:rustc-link-lib=cudart`.
2. PTX is embedded in the binary via `include_bytes!` and loaded at runtime through cudarc.
3. `cargo build` without `--features cuda` is the default path and must not touch CUDA.

## Work queue (first PR set on `gpu-device-abstraction`)

- [ ] (Done) `Device` enum, `Backend` trait, `NdArrayBackend` pass-through. Main stays CPU-pure.
- [ ] Add cudarc as an optional dep gated on `cuda`.
- [ ] `build.rs` detecting CUDA toolchain; no-op if absent.
- [ ] `CudaBackend` skeleton implementing `Backend` — every method returns a "not implemented" error.
- [ ] Kernel #1 (matmul via cuBLAS) + parity test. Merge when green on an actual GPU.
- [ ] Kernels #2-#9 one PR each, each with a parity test and a micro-benchmark.
- [ ] `scaled_dot_product_attention` parity test (composite).
- [ ] GPT-2 `generate()` running end-to-end on CUDA, asserted against the CPU path.
- [ ] Flash Attention v2 replacement for the naive attention composition.

## Non-goals

- **Training on GPU** — inference first. Training requires kernel derivatives, optimizer kernels, and correctness under NaN/Inf that inference doesn't stress.
- **Multi-GPU** — single-device for the first round.
- **Mixed precision** — f32 first, then bf16 once the kernels are stable.
- **cuDNN** — only if a specific kernel (dropout, batchnorm) is easier via cuDNN than writing our own. Prefer in-tree kernels for control.
