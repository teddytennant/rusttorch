# RustTorch TODO Queue

Durable task queue for the dedicated rusttorch Ralph loop (`/home/gradient/rusttorch-loop.sh`). Ralph picks the topmost unchecked task, implements it with tests, runs `cargo fmt && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace`, commits, pushes, checks the box, and logs.

## Ground rules

- Every commit must leave `cargo test --workspace` green and `cargo clippy --workspace --all-targets -- -D warnings` clean.
- Git author: `Teddy Tennant <teddytennant@icloud.com>`. No Co-Authored-By lines.
- Do not batch tasks — one logical unit per commit.
- If a task is ambiguous or blocked, check the box with a `(skipped: reason)` note and move on.
- For tasks that need GPU access, leave them for later and skip.

## Track E — GPU device abstraction (branch `gpu-device-abstraction`)

The rest of the tracks go on `main`. These tasks go on a dedicated branch so main stays stable until the refactor lands cleanly.

- [ ] Branch `gpu-device-abstraction` from main.
- [ ] Add `tensor::Device` enum with variants `CPU` and `CUDA(usize)`. CUDA variant behind a `cuda` cargo feature.
- [ ] Add a `cuda` feature flag in `rusttorch-core/Cargo.toml` (no dep yet).
- [ ] Add `Backend` trait in `rusttorch-core/src/backend/mod.rs` with methods: `allocate`, `copy_h2d`, `copy_d2h`, `matmul`, `add`, `mul`, `relu`, `softmax`, `layernorm`, `embedding_lookup`, `gelu`.
- [ ] Implement `NdArrayBackend` (CPU) in `rusttorch-core/src/backend/ndarray_backend.rs`. Migrate ops module-by-module, keeping all tests green.
- [ ] Thread `device: Device` through `Tensor::zeros`, `Tensor::ones`, `Tensor::from_vec`. Default to `Device::CPU` via a `*_on_device` variant so existing call sites keep compiling.
- [ ] Add `tensor.to(Device::CUDA(0))` stub method returning an error when the `cuda` feature is off.
- [ ] Write `GPU_KERNELS.md` at repo root: list every CUDA kernel needed for GPT-2 end-to-end (matmul, gelu, layernorm, softmax, embedding lookup, attention QKV reshape, top-k sampling). For each: signature, expected bandwidth/throughput, implementation notes.
- [ ] Once the refactor is stable on the branch, merge it to main in a single PR.

## Track F — Port recent PyTorch features

- [ ] Add `RMSNorm` module (`rusttorch-core/src/nn/rmsnorm.rs`). Tests: forward against a reference, gradcheck.
- [ ] Add `GroupNorm` module. Tests: forward + gradcheck.
- [ ] Add `SwiGLU` activation (Llama-style: `x * silu(gate)`). Tests: forward + gradcheck.
- [ ] Add `scaled_dot_product_attention` fused op to `nn::attention`. Wire it into the existing `MultiHeadAttention` path as an opt-in.
- [ ] Add `torch.func`-style `vmap` over a scalar function via rayon. Put it in `rusttorch-core/src/func/mod.rs`. Tests: confirm equivalence with a manual Python-style for-loop over the batch dim.
- [ ] Add `torch.func`-style `grad` (returns a function that computes `df/dx`). Tests: match finite-difference.
- [ ] Add `torch.func`-style `jacrev` (reverse-mode Jacobian). Tests: tiny 2x2 quadratic.
- [ ] Dataloader prefetch: rayon + bounded channel. Benchmarks vs unprefetched.
- [ ] Write `PYTORCH_PORT.md` tracking what was ported, skipped, and why.

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

- [x] Expose autograd/Linear/SGD/Adam/MSELoss to Python (2026-04-12)
- [x] AlignedBuffer + BumpArena memory pool (2026-04-12)
- [x] Foundational tests (dtype, shape, storage, view, error, utils, memory) and integration gradcheck + end-to-end training suites — 648 total tests (2026-04-12)
- [x] Full clippy -D warnings cleanup, plus fix four correctness errors including two dead test assertions (2026-04-12)
