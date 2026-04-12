# RustTorch TODO Queue

Durable task queue for the dedicated rusttorch Ralph loop (`/home/gradient/rusttorch-loop.sh`). Ralph picks the topmost unchecked task, implements it with tests, runs `cargo fmt && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace`, commits, pushes, checks the box, and logs.

## Ground rules

- Every commit must leave `cargo test --workspace` green and `cargo clippy --workspace --all-targets -- -D warnings` clean.
- Git author: `Teddy Tennant <teddytennant@icloud.com>`. No Co-Authored-By lines.
- Do not batch tasks — one logical unit per commit.
- If a task is ambiguous or blocked, check the box with a `(skipped: reason)` note and move on.
- For tasks that need GPU access, leave them for later and skip.

## Track E — GPU device abstraction (branch `gpu-device-abstraction`)

First slice shipped on the branch — main stays CPU-pure until it lands cleanly.

- [x] Branch `gpu-device-abstraction` from main.
- [x] Add `tensor::Device` enum with variants `Cpu` and `Cuda(usize)`. CUDA variant behind a `cuda` cargo feature.
- [x] Add a `cuda` feature flag in `rusttorch-core/Cargo.toml` (no dep yet).
- [x] Add `Backend` trait in `rusttorch-core/src/backend/mod.rs`. Minimal surface covering the GPT-2 inference critical path: matmul, add, mul, relu, gelu, log_softmax, layer_norm, scalar.
- [x] Implement `NdArrayBackend` (CPU) in `rusttorch-core/src/backend/ndarray_backend.rs` as a pass-through over existing ops.
- [x] Thread `device: Device` through every `Tensor` constructor. Default to `Device::Cpu` so every existing caller keeps compiling unchanged. Added `.device()`, `.to(Device)`, `.cpu()`, `.cuda(id)` accessors.
- [x] Add `tensor.to(Device::Cuda(0))` stub method returning a clear error when the `cuda` feature is on but kernels haven't shipped.
- [x] Write `GPU_KERNELS.md` at repo root — full kernel queue, dependency plan (cudarc + cuBLAS), correctness strategy, perf targets for A100.
- [ ] Replace `TensorData` enum with a Backend-owned `Storage` associated type so GPU buffers don't round-trip through the ndarray enum. (Follow-up refactor, larger scope.)
- [ ] Migrate ops module-by-module through the Backend trait. (Incremental; can happen alongside step above.)
- [ ] Add cudarc as an optional dep gated on `cuda`. `build.rs` detects CUDA toolchain and no-ops if absent.
- [ ] Implement the first real CUDA kernel (`matmul` via cuBLAS) + CPU/GPU parity test. Needs GPU access.
- [ ] Kernels #2–#9 from `GPU_KERNELS.md` Tier 1 in order. Each one PR, each with a parity test and a micro-benchmark.
- [ ] Once the refactor is stable on the branch, merge it to main in a single PR.

## Track F — Port recent PyTorch features

- [x] Add `RMSNorm` module (`rusttorch-core/src/nn/rmsnorm.rs`). Tests: forward against a reference, gradcheck.
- [x] Add `GroupNorm` module. Tests: forward + gradcheck.
- [x] Add `SwiGLU` activation (Llama-style: `silu(gate) * value`). Tests: forward + gradcheck for both inputs.
- [x] Add `SiLU` activation module and `Variable::silu()` primitive.
- [x] Add `scaled_dot_product_attention` helper mirroring the PyTorch free function, `is_causal` flag included.
- [x] Write `PYTORCH_PORT.md` tracking what was ported, skipped, and why.
- [ ] Add `torch.func`-style `vmap` over a scalar function via rayon. Blocked on: autograd-tracked `slice` and `concat`/`stack` ops don't exist yet. Adding those is the prerequisite.
- [ ] Add `torch.func`-style `grad` (returns a function that computes `df/dx`). Tests: match finite-difference.
- [ ] Add `torch.func`-style `jacrev` (reverse-mode Jacobian). Tests: tiny 2x2 quadratic.
- [ ] Dataloader prefetch: rayon + bounded channel. Benchmarks vs unprefetched.
- [ ] Add autograd-tracked `slice_along_dim` and `concat_along_dim` ops. (Prerequisite for vmap and the chunking operations.)

## Track B follow-ups — more foundational tests

- [ ] Add property-based tests in `tensor/shape.rs` using the `proptest` dev-dep (broadcast invariants, stride roundtrip).
- [ ] Add a `Storage::as_slice` aliasing test under Miri once the workspace runs under Miri cleanly.
- [ ] Add an integration test that loads a safetensors file, runs one forward pass, and saves a fresh state_dict, verifying roundtrip invariance.

## Track C follow-ups — richer Python bindings

- [ ] `PyConv2d`, `PyBatchNorm2d`, `PyLayerNorm`, `PyDropout`.
- [ ] `PySequential` taking a Python list of modules. Likely needs boxed trait objects with a custom PyO3 trait.
- [ ] `PyCrossEntropyLoss`.
- [ ] `PyAdamW`, `PyStepLR`, `PyCosineAnnealingLR`.
- [ ] Add `tensor.grad` property (not method) for PyTorch compat.
- [ ] Stub type hints in `python/rusttorch/__init__.pyi`.

## Track D follow-ups — memory module

- [ ] Criterion benchmark comparing a training step using `BumpArena` for temporaries vs one using plain Vec allocation.
- [ ] Thread `BumpArena` through one op as a proof-of-concept (e.g., `layer_norm_forward`), gated behind a `pool_scratch` feature flag so it doesn't affect existing callers.

## Examples and polish

- [ ] Write an `examples/xor_py.py` that mirrors the Rust `mlp_learns_xor` test, as a runnable tutorial.
- [ ] Add a CI config (`.github/workflows/ci.yml`) running fmt, clippy -D warnings, test, and the maturin build.
- [ ] Clean up `.claude/plan.md` to reflect the current state of the repo (many items are now done).
- [ ] README tidy pass: the "Alpha" status is out of date given the test count and feature surface.

## Completed (most recent first)

- [x] SDPA top-level helper and `PYTORCH_PORT.md` — 672 total tests (2026-04-12)
- [x] GroupNorm module + autograd op + 8 unit tests + gradcheck (2026-04-12)
- [x] SiLU activation + SwiGLU helper + 3 gradcheck tests (2026-04-12)
- [x] RMSNorm module + autograd op + 7 unit tests + 3 gradcheck tests (2026-04-12)
- [x] GPU prep branch: Device enum, Backend trait, NdArrayBackend pass-through, cuda feature flag, GPU_KERNELS.md spec (on `gpu-device-abstraction`) (2026-04-12)
- [x] RUSTTORCH_TODO.md + rusttorch-loop.sh + brain entry (Track G) (2026-04-12)
- [x] Expose autograd/Linear/SGD/Adam/MSELoss to Python — 648 total tests (2026-04-12)
- [x] AlignedBuffer + BumpArena memory pool — 648 tests (2026-04-12)
- [x] Foundational tests (dtype, shape, storage, view, error, utils, memory) and integration gradcheck + end-to-end training suites — 630 tests (2026-04-12)
- [x] Full clippy -D warnings cleanup, plus fix four correctness errors including two dead test assertions (2026-04-12)
