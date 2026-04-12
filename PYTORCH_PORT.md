# PyTorch Feature Port Tracker

Running log of PyTorch 2.x features ported to rusttorch, skipped, or deferred. This is Track F in the repo plan (`.claude/plans/lucky-stargazing-spindle.md`) and gets updated whenever a PyTorch feature lands.

## Ported

### Normalization

- **RMSNorm** — `nn::RmsNorm`. Root-mean-square layer norm (Zhang & Sennrich 2019). Used in Llama/Mistral/Gemma. Cheaper than LayerNorm (no mean subtraction, no bias), equally effective on LLM workloads. Forward reference match, gradcheck against both input and weight.
- **GroupNorm** — `nn::GroupNorm`. Per-sample per-group normalization (Wu & He 2018). Works with batch size 1, unlike BatchNorm. Degenerates to LayerNorm when `num_groups=1` and to InstanceNorm when `num_groups=num_channels`. Forward verified against both limits, backward gradcheck passes.

### Activations

- **SiLU / Swish** — `nn::SiLU` module and `Variable::silu()`. Composed from existing `sigmoid` and `mul` primitives so autograd is free. `y = x * sigmoid(x)`. Gradcheck passes.
- **SwiGLU** — `nn::swiglu(gate, value)` free function. The element-wise step inside Llama-style gated FFN blocks: `silu(gate) * value`. Free function (not a `Module`) because it takes two inputs; caller builds the full FFN with separate gate/value/down projections. Gradcheck passes for both inputs.

### Attention

- **`scaled_dot_product_attention`** — `nn::scaled_dot_product_attention(q, k, v, is_causal)`. Thin wrapper mirroring `torch.nn.functional.scaled_dot_product_attention`. The underlying op (`autograd::ops::scaled_dot_product_attention_forward` / `_causal_forward`) already existed and handles backward; this just gives callers the top-level API without importing `autograd::ops`. Integration tests for causal and non-causal paths.

## Deferred (too big for a single pass, will come back)

- **`torch.func` functional API** — `vmap`, `grad`, `jacrev`, `jacfwd`, `hessian`. `vmap` is the highest-value one and maps cleanly onto rayon parallelism, but a gradient-preserving implementation needs autograd-tracked slice and stack ops that don't exist yet. A future pass will add those first, then build `vmap` on top.
- **Memory-efficient attention path** — Flash Attention 2 style fused kernel. CPU version is bandwidth-bound so gains are modest; real win is on GPU, so this is deferred to the `gpu-device-abstraction` branch.
- **Dataloader prefetch** — `rayon` + bounded channel, mirroring PyTorch `DataLoader`'s `prefetch_factor` / `persistent_workers`. Useful but scoped separately from the activation/normalization port.
- **`F.rms_norm` free function** — currently you call the method on `Variable` or the `RmsNorm` module. A free function mirroring PyTorch's signature is cosmetic and low priority.
- **`nn.utils.parametrize`** — requires a trait-object rewrite of `Module`, not a feature addition. Skipped.

## Skipped (not feasible)

- **`torch.compile` / Dynamo / Inductor** — needs a full graph compiler backend. Out of scope for a hand-written library without LLVM integration.
- **Per-device RNG** — rusttorch uses a single global RNG from the `rand` crate. PyTorch's per-device RNG is useful for GPU training but unnecessary here.
- **NestedTensor / jagged batches** — large API surface, niche use case.

## Notes for future work

- Every normalization op so far uses the "save intermediate + derive closed-form backward" pattern from `LayerNormBackward`. If this keeps up, extracting a small macro to generate the `GradFn` boilerplate might be worth it — tracking TODO in `RUSTTORCH_TODO.md`.
- The autograd surface could use a `slice` and `concat` op pair; once they exist, `vmap`, `chunk`, `stack`, and a better `split` all become trivial. Likely the highest-leverage single addition.
- For `scaled_dot_product_attention`, the CPU impl is currently the three-step Q@K^T → softmax → @V chain. A fused CPU kernel is not worth writing since the GPU one will obsolete it.
