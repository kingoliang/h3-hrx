# Install and run

h3-hrx runs MiniMax H3 on Linux with AMD Strix Halo (`gfx1151`). Development and
[measured renders](showcase.md#wyvern-chase--anime-and-cinematic) use a 128 GB
unified-memory system. Other GPUs are unvalidated.

## Dependencies

- A working AMD driver and GPU access.
- A current stable Rust toolchain and ffmpeg.
- System libraries compatible with HRX's pinned native bundle: Ubuntu 26.04,
  glibc 2.43+, or a compatible environment. See
  [HRX native setup](https://github.com/zacharydenton/hrx-rs/blob/main/docs/GPU-NPU.md#native-setup).
- About 54 GB of disk space for base checkpoints, plus runtime caches and outputs.

HRX supplies the Loom compiler and GPU runtime. Building h3 requires no ROCm
headers, `hipcc` or local LLVM build.

## Toolchain and build

From the repository root:

```sh
cargo install --locked --path . --bin h3
h3 --width 1344 --height 768 --frames 124 --steps 21 --seed 0 \
  --weight-io native-direct --memory-budget-mib 49152 \
  --out clip.mp4 < docs/prompts/wyvern_cinematic.txt
```

This writes `clip.mp4` and `clip.wav`. First use fetches missing checkpoints and
HRX's verified native bundle, then compiles the Loom kernels. `--steps 21` gives
20 model evaluations. For development, use `cargo build --locked --release --bin h3`.

The default CLI canvas is 864×480 with 124 frames and 30 evaluations. The command
above selects the featured 768p workload. Dimensions must be multiples of 32;
frame counts round up to `17n + 5`. See [prompts](prompting.md) and
[runtime options](runtime-options.md) for inputs, memory limits and loading modes.

## Offline runtime

Provision the HRX version used by this checkout before going offline:

```sh
cargo install --locked hrx-rs --version 0.10.1
hrx prepare
hrx doctor
```

Alternatively, supply that version's matching native archive to
`HRX_OFFLINE=1 hrx prepare native.tar.gz`. Set `HRX_OFFLINE=1` to prevent native
bundle downloads. Checkpoint downloads have a separate offline switch below.
[Runtime overrides and cache locations](shared-hrx.md#provisioning-and-overrides).

## Checkpoints

Missing checkpoints download from
[Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3).
No conversion step is needed.

| Checkpoint | Contents | Disk size |
| --- | --- | ---: |
| `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | Base DiT | ~21 GB |
| `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | Text encoder and vision tower | ~27 GB |
| `minimax_h3_video_vae_fp16.safetensors` | Video encoder and decoder | ~5.2 GB |
| `minimax_h3_audio_vae_fp32.safetensors` | Audio encoder and decoder | ~0.6 GB |
| `minimax_h3_ref2va_pruned_int8_convrot.safetensors` | Reference-conditioned DiT | ~21 GB additional |

The default cache is `~/.cache/huggingface/hub`. Precedence is `HF_HUB_CACHE`,
then `$HF_HOME/hub`, then `$XDG_CACHE_HOME/huggingface/hub`.
`--offline` or `HF_HUB_OFFLINE=1` requires cached checkpoints.
Use `--dit`, `--te`, `--video-vae` or `--audio-vae` for explicit file paths.
Model weights have separate license terms.

Custom H3 merges can mix BF16 and INT8 ConvRot transformer blocks, including
INT8 token refiners and condition/output projections. Each block's four matrices
must share a dtype; INT8 matrices require per-row FP32 scales and ConvRot group
size 256. BF16/F16/FP32 embedding weights, biases and normalization vectors are
accepted. The small condition/output projections are dequantized and unrotated
when loaded; transformer blocks run in their stored precision.

For example, use a local merged checkpoint with the cached text encoder and VAEs:

```sh
cargo run --locked --release --bin h3 -- \
  --dit ~/h3_b6_int8.safetensors --offline \
  --width 512 --height 288 --frames 22 --steps 21 \
  --out clip.mp4 < prompt.txt
```

Reference requests select Ref2VA; text and keyframe-only requests select FL2VA.
`--base-weights` forces FL2VA with references. To prefetch files with the optional
Hugging Face CLI:

```sh
hf download Comfy-Org/MiniMax-H3 \
  --include '*minimax_h3_fl2va_pruned_int8_convrot.safetensors' \
            '*qwen3vl_32b_minimax_h3_int8_convrot.safetensors' \
            '*minimax_h3_video_vae_fp16.safetensors' \
            '*minimax_h3_audio_vae_fp32.safetensors'
# Additional model for references:
hf download Comfy-Org/MiniMax-H3 \
  --include '*minimax_h3_ref2va_pruned_int8_convrot.safetensors'
```

### FastH3 sampling

Use an explicit compatible checkpoint with the FastH3 preset:

```sh
h3 --dit ~/h3_b6_int8.safetensors --preset fasth3 --evaluations 10 -p 'Your prompt'
```

The preset selects `res_multistep`, the ComfyUI `simple` scheduler, video/audio
shifts 10/3, and ten model evaluations. Explicit `--sampler`, `--scheduler`,
`--video-shift`, `--audio-shift`, and `--evaluations` override those values.
Use 8–12 evaluations to explore the recommended range. `--steps` retains its
legacy meaning (sigma points, one more than evaluations) and cannot be combined
with `--evaluations`. Existing defaults stay unchanged. No filename detection
or alternate model download is performed.

Rust callers use `SamplingOptions::resolve(params, config.dit.is_some())`, with
`preset: Some(SamplingPreset::FastH3)` and optional overrides. Simple scheduling
matches the pinned ComfyUI FP32 table and index selection; its audio schedule
remaps the selected video sigmas to the audio shift. This is schedule parity,
not a claim of bitwise end-to-end ComfyUI parity.

Schedule fixtures come from ComfyUI revision
`7a28332047b6af638dc6c7f72ef3481a718769fd`; refresh them with
`python scripts/fasth3_schedule_fixture.py` (requires PyTorch and network access).

For VSA, add `--attention-pattern vsa`. Attention precision remains a separate
choice (`--attn i8` or `--attn f16`). The default keeps 10% of non-prefix blocks,
starts at sampling progress 0.2, ends at 1, and uses dense attention below 12,288
tokens. Those progress values are converted to sigma thresholds using the video
shift. Small test clips therefore need `--vsa-min-tokens 0` to exercise VSA.

```sh
h3 --dit ~/h3_b6_int8.safetensors --preset fasth3 --evaluations 12 \
  --attention-pattern vsa --width 1344 --height 768 --frames 124 -p 'Your prompt'
```

`--vsa-keep-percent`, `--vsa-start`, `--vsa-end`, `--vsa-min-tokens`, and
`--vsa-dense-blocks 0,49` override the sparse policy. Block indices are global,
zero-based DiT indices, including checkpoints with mixed weight precision.
The preset itself defaults to dense attention. VSA is supported with gate-free
B6 and compatible H3 checkpoints carrying `attn.to_gate_compress` weights.
INT4, World, Turbo, and residual step-cache reuse are rejected with VSA; cache
observation is supported. Text encoding, token refinement, VAEs, and ER-SDE
upscale refinement use dense attention.

Rust configuration is equivalent:

```rust
use h3_hrx::{AttentionPattern, Config, DenoiseParams, SamplingOptions,
             SamplingPreset, VsaOptions};
let config = Config {
    dit: Some("/path/to/h3_b6_int8.safetensors".into()),
    attention_pattern: AttentionPattern::Vsa(VsaOptions::default()),
    ..Default::default()
};
let params = SamplingOptions {
    preset: Some(SamplingPreset::FastH3),
    evaluations: Some(12),
    ..Default::default()
}.resolve(DenoiseParams::default(), config.dit.is_some())?;
```

See [FastH3 validation](fasth3.md) for reference revisions, numerical checks,
qualification commands, and the distinction between algorithm and kernel parity.
