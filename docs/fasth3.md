# FastH3 sampling and VSA

The explicit `fasth3` preset selects 10/3 shifts, `simple`, `res_multistep`, and
10 evaluations. Existing defaults and the legacy sigma-point `steps` field are
preserved. Euler is also supported; no additional samplers or SLA/Sol modes are
introduced. The checkpoint must have the existing H3 architecture and folded
AdaLN tables.

## Reference definitions

The implementation and reproducible fixtures use:

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI/tree/7a28332047b6af638dc6c7f72ef3481a718769fd),
  revision `7a28332047b6af638dc6c7f72ef3481a718769fd`.
- [comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen/tree/f8b7e97a1418d2e529bd5405fbf86312be198ae6),
  revision `f8b7e97a1418d2e529bd5405fbf86312be198ae6`.
- [FastH3 workflow](https://github.com/Comfy-Org/workflow_templates/blob/447747bf37b7e2f6d5fab3f15288e03bf889b82f/templates/video_fastvideo_fasth3_t2v.json),
  revision `447747bf37b7e2f6d5fab3f15288e03bf889b82f`.
- [Gated FastH3 checkpoint](https://huggingface.co/FastVideo/FastVideo-FastH3-Comfy/tree/ec1e3aa374a91c57b0b94a1623b7e657c0498cf2),
  revision `ec1e3aa374a91c57b0b94a1623b7e657c0498cf2`.

`simple` selects entries from the 1,000-entry FP32 flow table using ComfyUI's
index rule, then remaps video sigmas to the audio shift. Tests compare both
streams and their clean-time conditioning values bit-for-bit, including 8, 10,
12, non-divisor counts and boundaries. The legacy uniform calculation is unchanged.

VSA packs video into 4×4×4 cubes, stably puts live rows before padding, and pads
each prefix segment independently to 64 rows. Prefix keys and queries remain
exact. Routing selects the top fraction of non-prefix blocks per head, retains
ties, and forces the diagonal ±1 blocks. Padding never contributes as a key.
There is no pooled tail or extra-token augmentation.

Compatible learned compression gates add dense attention over the block means,
multiplied by the raw linear gate; there is no sigmoid. Gate weights accept BF16,
F16, F32, or validated per-row INT8 ConvRot H256. They are prepared as BF16 for
the native gate projection. B6 has no such gates and uses fine attention alone.

This is native algorithm compatibility, not bitwise ComfyUI kernel parity.
Routing uses FP32 means. INT8 attention quantizes rotated Q/K while retaining
FP16 V; it reuses the previous sparse evaluation's per-block K-centering state.
Unlike the upstream all-INT8 producer, it needs no stale V scale or V clipping.
All state resets between requests. The existing native activation arithmetic
also differs from ComfyUI's BF16 path.

GPU gather, pooling, routing, sparse attention, optional coarse attention, and
scatter execute without host routing readbacks. Only block-by-block score and
route matrices are allocated. Dense and sparse recordings are separate; every
replay recomputes routes from the current Q/K. Sequential mixed-precision runs
share scratch buffers, with graph dependencies preserving the ordering.

## Reproduce validation

CPU/API checks:

```sh
cargo test --locked --release --workspace --no-default-features --features cli,prompt-generation
cargo clippy --workspace --all-targets --all-features -- -D warnings
```

Native kernel and pinned PyTorch oracle checks:

```sh
H3_VSA_FIXTURE=target/vsa-fixture cargo test --release --lib vsa::device::tests -- --nocapture
python scripts/vsa_reference.py target/vsa-fixture
```

The native test covers partial cubes, exact prefixes, tied routing scores,
negative compression gates, both precision modes, and eager/graph equality.
Thresholds are normalized RMSE ≤1% for FP16 and ≤3% for INT8. On gfx1151 the
pinned oracle comparison measured approximately 0.014% and 0.18%, respectively,
for both fine-only and gated fixtures.

The matrix harness prepares text once, releases the encoder, and then runs
Euler and res_multistep at 8, 10, and 12 evaluations. It checks finite outputs,
cancellation after sparse work, layout changes, deterministic retries, and
budget retirement.
Use `dense`, `vsa`, or `observe`; the latter two force VSA below its normal token
threshold so small fixtures actually exercise sparse kernels.

```sh
H3_GRAPH=1 H3_STAGE_TRACE=1 \
H3_QUALIFY_CHECKPOINT="$HOME/h3_b6_int8.safetensors" \
H3_QUALIFY_OUT=target/b6-matrix H3_QUALIFY_PATTERN=vsa \
cargo test --release --lib dit::qualification::fasth3_matrix -- --ignored --exact --nocapture
```

Repeat with `H3_GRAPH=0` to compare eager output files byte-for-byte, and with
`H3_QUALIFY_PATTERN=observe` to compare cache observation. For production-size
qualification add `H3_QUALIFY_FULL=1`: this runs 1344×768, 124 frames, ten
evaluations after a two-evaluation warmup and restores the normal
12,288-token threshold. Step timings and native reserved memory are printed separately
from model preparation. `H3_STAGE_TRACE=1` records all resolved sampling and
attention options, sigmas, evaluation counts, sparse-block counts, and dense
fallback reasons.

Legacy digest baselines are not regenerated by this work. The existing standard
checkpoint denoising digest test already failed on the original checkout:
`shortest video` produced
`76fd78fcc2ceab0ea5ba179acf421715816b4a9f2ad21bfebd19513c6a0abfdd`
instead of the frozen
`c64edbf4aa8c59a40a75378a6e49cce2af15dc353ab71aa9fbe5ed53dcd2c06c`.
That baseline issue remains unresolved. The final recheck was interrupted when
the shared GPU was needed for other work.

## Model qualification

The six-case sampler matrix passed on gfx1151 for the mixed BF16/INT8,
gate-free B6 checkpoint in dense and VSA modes, and for the pinned FastH3
checkpoint with all 50 learned compression gates in VSA mode. All video and
audio latent values were finite, retries were deterministic, and each test
returned its native memory budget to zero after teardown.

B6 VSA with cache observation produced byte-identical video and audio latents
to VSA with caching off in all six cases. Sharing scratch between sequential
mixed-precision groups also preserved all six outputs byte-for-byte. Eager and
graph execution produced byte-identical video and audio latents for all six
B6 VSA cases. Both runs also passed cancellation and layout-change retry checks.

Full-size qualification also exercises the large dense fallback before entering
VSA. Its plain V transpose clears the complete reused capacity: the large
attention kernel can read a 64-key tile beyond the last 32-row transpose tile,
and masked padding must contain zeros rather than stale activation bits.

For isolated device timing, run the ignored
`vsa::device::tests::full_size_profile` test with `H3_PROFILE=device`.
The model matrix also accepts `H3_PROFILE=device` and
`H3_QUALIFY_EVALUATIONS=2` for a short diagnostic run. Add
`H3_VSA_DIAGNOSTICS=1 H3_GRAPH=0` to assert finite first-block pooled Q/K values
and print route density. These diagnostic readbacks exist only in test builds.

### Full-resolution result

A sequential gfx1151 comparison used B6, INT8 Q/K attention, 1344×768,
124 frames, seed 7, and the prompt “A red ball rolls across a wooden table.”
Both runs used `res_multistep`, `simple`, 10/3 shifts, ten evaluations, and a
two-evaluation warmup. Video and audio latents were finite in both modes, and
both tests retired their complete native memory budgets.

| Attention | Ten-evaluation sampler time | Native reserved memory |
| --- | ---: | ---: |
| Dense | 929.8 s | 27.0 GiB |
| VSA, default policy | 959.8 s | 32.2 GiB |

VSA used dense attention for the first two evaluations and sparse attention in
all 50 blocks for the remaining eight. In this measurement it was about 3.2%
slower and reserved about 5.2 GiB more memory. This implementation provides the
VSA algorithm and sampling controls, but does not demonstrate acceleration on
this GPU. Dense remains the default. These are single-run sampler measurements
on a shared development machine; they exclude checkpoint preparation, text
encoding, and VAE decoding, and do not
establish a general quality or performance ranking.

The raw outputs are under `target/b6-full-dense/` and `target/b6-full-vsa/`.
Decode each with the normal native VAEs:

```sh
cargo run --release --example render_latents -- \
  target/b6-full-vsa/ResMultistep-10 1344 768 124 target/b6-fasth3-vsa.mp4
```

Full-resolution VAE decoding and visual/audio inspection were deferred to avoid
additional load on the shared GPU. The finite-latent checks and sampler timings
above are complete; they are not an end-to-end media-quality qualification.
