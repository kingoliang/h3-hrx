# H3 generation from Elixir

This Rustler 0.38 example calls H3's Rust API and returns results as BEAM messages.
A dedicated worker owns each session and accepts jobs through a bounded queue,
keeping inference off ordinary schedulers.

Build from the repository root and check worker startup and request validation:

```sh
cargo build --manifest-path clients/rustler/Cargo.toml
H3_NIF="$PWD/target/debug/libh3_nif" elixir clients/rustler/smoke.exs
```

## Audio and generation

The example exports `open/1`, `ping/2`, `encode_audio/4` and `generate/3`. For audio, pass a
checkpoint path to `open/1`, then an interleaved little-endian float32 binary,
frame count, and request ID with the model handle to `encode_audio/4`. Results arrive as
`{:h3_result, id, {:ok, {latents, frames}}}` or `{:h3_result, id, {:error, reason}}`.
The example returns latents as a list; an application handling large outputs
should encode them as an owned binary. A full queue rejects the request.

`generate/3` queues generation, learned upscaling and ER-SDE refinement:

```elixir
:ok = H3.Native.generate(model, %{
  prompt: File.read!("docs/prompts/wyvern_cinematic.txt"), width: 640, height: 384,
  frames: 124, steps: 21, seed: 0, references: [], refmods: [],
  upscale_width: 1344, upscale_height: 768, upscale_steps: 4,
  upscale_denoise: 0.4, upscale_seed: nil
}, 43)
```

Reference entries are `{:image, latents, latent_height, latent_width}`,
`{:audio, latents, audio_frames}` or
`{:video, latents, latent_frames, latent_height, latent_width}`. Float lists use
the native Session layouts. RefMod entries are
`{path, visual_strength, audio_strength, copies}`. They use latent-only
presentation; the same prepared conditioning reaches both passes.

Generation returns `{:h3_result, id, {:ok, {width, height, frames, video, audio}}}`.
Video and audio are little-endian float32 binaries of normalized latents; audio
is unchanged by the second pass. Decode using the native Session API in an
application adapter. Models resolve through the Hub cache. The worker uses
stage-scoped residency.

## Worker ownership

NIF inputs are copied into owned job data before returning. The worker releases
its GPU session after the resource is collected. Keep model files immutable while
the worker lives. Rustler is confined to this adapter.

[Rustler resources](https://docs.rs/rustler/0.38.0/rustler/struct.ResourceArc.html) ·
[NIF scheduling](https://docs.rs/rustler/0.38.0/rustler/attr.nif.html)

## Explicit FastH3 sampling

`open_with_options/1` accepts an atom-keyed map. Omitted keys keep defaults:
`dit`, `te`, `video_vae`, `audio_vae` (path strings), and `attention`
(`"f16"`, `"i8"`, or `"i4"`). Unknown keys are rejected.

`generate_with_options/4` accepts `(model, request, options, id)`, retaining the
existing request and result formats. Options include `preset: "fasth3"`,
`evaluations: 10`, `scheduler: "simple"`, `sampler: "res_multistep"` or `"euler"`,
`video_shift: 10.0`, `audio_shift: 3.0`, and `bypass_upscale: true`.
The FastH3 preset requires a model opened with an explicit `dit` path.
Explicit options override the preset. `steps` in options is a legacy sigma-point
count and conflicts with `evaluations`; either overrides the request's `steps`.
With neither option nor preset, the request's `steps` is retained.
Bypassing upscaling also bypasses refinement and validation of upscale fields.
The original `open/1` and `generate/3` remain unchanged.

```elixir
model = H3.Native.open_with_options(%{dit: Path.expand("~/h3_b6_int8.safetensors")})
:ok = H3.Native.generate_with_options(model, request,
  %{preset: "fasth3", evaluations: 12, bypass_upscale: true}, 44)
```

`open_with_options/1` also accepts `attention_pattern: "vsa"`,
`vsa_keep_percent`, `vsa_start`, `vsa_end`, `vsa_min_tokens`, and
`vsa_dense_blocks` (a list of zero-based global DiT indices). Defaults match the
CLI. VSA options require `attention_pattern: "vsa"`; VSA rejects `attention: "i4"`.
These are session options; sampling options belong to `generate_with_options/4`.
