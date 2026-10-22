//! App-owned Rustler adapter, calling H3's Rust API directly. A worker owns each
//! model session so GC, inference and model destruction never occupy a normal
//! BEAM scheduler. Jobs contain owned data, never a borrowed NIF environment.
use rustler::{Atom, Binary, Encoder, Env, LocalPid, NifResult, OwnedEnv, ResourceArc};
use std::sync::mpsc::{self, SyncSender};
mod generate;
mod options;
mod atoms {
    rustler::atoms! { ok, h3_result }
}
enum Job {
    Generate(
        LocalPid,
        u64,
        generate::GenerateRequest,
        options::GenerateOptions,
    ),
    Ping(LocalPid, u64),
    Encode(LocalPid, u64, Vec<f32>, usize),
}
struct Model {
    jobs: SyncSender<Job>,
}
#[rustler::resource_impl]
impl rustler::Resource for Model {}

/// Checkpoint files supplied by the application must remain unchanged for the
/// resource's lifetime, as required by the model's mmap-based Rust API.
#[rustler::nif(schedule = "DirtyIo")]
fn open(audio_checkpoint: Option<String>) -> NifResult<ResourceArc<Model>> {
    let config = h3_hrx::Config {
        dit: None,
        te: None,
        video_vae: None,
        audio_vae: audio_checkpoint.map(Into::into),
        ..Default::default()
    };
    open_config(config)
}

#[rustler::nif(schedule = "DirtyIo")]
fn open_with_options(options: rustler::Term<'_>) -> NifResult<ResourceArc<Model>> {
    let mut o = options::Options::new(options)?;
    let attention = match o.get::<String>("attention")?.as_deref() {
        None | Some("i8") => h3_hrx::Attention::I8,
        Some("f16") => h3_hrx::Attention::F16,
        Some("i4") => h3_hrx::Attention::I4,
        _ => return Err(rustler::Error::BadArg),
    };
    let pattern_name = o.get::<String>("attention_pattern")?;
    let keep_percent = o.get("vsa_keep_percent")?;
    let start = o.get("vsa_start")?;
    let end = o.get("vsa_end")?;
    let min_tokens = o.get("vsa_min_tokens")?;
    let dense_blocks: Option<Vec<usize>> = o.get("vsa_dense_blocks")?;
    let attention_pattern = match pattern_name.as_deref() {
        Some("vsa") => h3_hrx::AttentionPattern::Vsa(h3_hrx::VsaOptions {
            keep_percent: keep_percent.unwrap_or(10.0),
            start: start.unwrap_or(0.2),
            end: end.unwrap_or(1.0),
            min_tokens: min_tokens.unwrap_or(12288),
            dense_blocks: dense_blocks.unwrap_or_default(),
        }),
        None | Some("dense")
            if keep_percent.is_none()
                && start.is_none()
                && end.is_none()
                && min_tokens.is_none()
                && dense_blocks.is_none() =>
        {
            h3_hrx::AttentionPattern::Dense
        }
        _ => return Err(rustler::Error::BadArg),
    };
    let config = h3_hrx::Config {
        dit: o.get::<String>("dit")?.map(Into::into),
        te: o.get::<String>("te")?.map(Into::into),
        video_vae: o.get::<String>("video_vae")?.map(Into::into),
        audio_vae: o.get::<String>("audio_vae")?.map(Into::into),
        attention,
        attention_pattern,
        ..Default::default()
    };
    o.finish()?;
    open_config(config)
}

fn open_config(config: h3_hrx::Config) -> NifResult<ResourceArc<Model>> {
    let explicit_checkpoint = config.dit.is_some();
    let (jobs, receiver) = mpsc::sync_channel(2);
    let (ready, opened) = mpsc::sync_channel(1);
    std::thread::Builder::new()
        .name("h3-model".into())
        .spawn(move || {
            // Session construction also belongs on this thread: BEAM dirty schedulers
            // have small native stacks, especially for unoptimized Rust builds.
            let mut session = match unsafe {
                h3_hrx::Session::new_with_options(
                    config,
                    h3_hrx::SessionOptions {
                        residency: h3_hrx::ResidencyPolicy::StageScoped,
                        ..Default::default()
                    },
                )
            } {
                Ok(session) => {
                    if ready.send(Ok(())).is_err() {
                        return;
                    }
                    session
                }
                Err(error) => {
                    let _ = ready.send(Err(error.to_string()));
                    return;
                }
            };
            while let Ok(job) = receiver.recv() {
                let job = match job {
                    Job::Generate(pid, id, request, options) => {
                        let result =
                            generate::run(&mut session, request, options, explicit_checkpoint)
                                .map_err(|e| e.to_string());
                        let mut env = OwnedEnv::new();
                        let _ = env.send_and_clear(&pid, |env| {
                            let result = result.map(|out| {
                                let video = float_binary(env, &out.latents.video);
                                let audio = float_binary(env, &out.latents.audio);
                                (
                                    out.shape.size().0,
                                    out.shape.size().1,
                                    out.shape.frames,
                                    video,
                                    audio,
                                )
                            });
                            (atoms::h3_result(), id, result).encode(env)
                        });
                        continue;
                    }
                    other => other,
                };
                let (pid, id, result) = match job {
                    Job::Generate(..) => unreachable!(),
                    Job::Ping(pid, id) => (pid, id, Ok((Vec::new(), 0usize))),
                    Job::Encode(pid, id, samples, frames) => (
                        pid,
                        id,
                        session
                            .encode_audio(&samples, frames)
                            .map_err(|e| e.to_string()),
                    ),
                };
                let mut env = OwnedEnv::new();
                // A dead caller does not invalidate model state or keep a NIF borrow.
                let _ =
                    env.send_and_clear(&pid, |env| (atoms::h3_result(), id, result).encode(env));
            }
            // Last resource is gone: close the channel and drop the GPU session here.
        })
        .map_err(|e| rustler::Error::Term(Box::new(e.to_string())))?;
    opened
        .recv()
        .map_err(|e| rustler::Error::Term(Box::new(e.to_string())))?
        .map_err(|e| rustler::Error::Term(Box::new(e)))?;
    Ok(ResourceArc::new(Model { jobs }))
}
#[rustler::nif]
fn ping(env: Env<'_>, model: ResourceArc<Model>, id: u64) -> NifResult<Atom> {
    model
        .jobs
        .try_send(Job::Ping(env.pid(), id))
        .map_err(|e| rustler::Error::Term(Box::new(e.to_string())))?;
    Ok(atoms::ok())
}
#[rustler::nif(schedule = "DirtyCpu")]
fn encode_audio(
    env: Env<'_>,
    model: ResourceArc<Model>,
    samples: Binary<'_>,
    frames: usize,
    id: u64,
) -> NifResult<Atom> {
    if !samples.len().is_multiple_of(4) {
        return Err(rustler::Error::BadArg);
    }
    let samples = samples
        .as_slice()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    model
        .jobs
        .try_send(Job::Encode(env.pid(), id, samples, frames))
        .map_err(|e| rustler::Error::Term(Box::new(e.to_string())))?;
    Ok(atoms::ok())
}
fn float_binary<'a>(env: Env<'a>, values: &[f32]) -> Binary<'a> {
    let mut bytes = rustler::OwnedBinary::new(values.len() * 4).expect("binary allocation");
    for (dst, value) in bytes
        .as_mut_slice()
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(values)
    {
        dst.copy_from_slice(&value.to_le_bytes());
    }
    bytes.release(env)
}

/// Queue a two-pass generation. Decode into owned fields away from ordinary BEAM schedulers.
#[rustler::nif(schedule = "DirtyCpu")]
fn generate(
    env: Env<'_>,
    model: ResourceArc<Model>,
    request: generate::GenerateRequest,
    id: u64,
) -> NifResult<Atom> {
    model
        .jobs
        .try_send(Job::Generate(env.pid(), id, request, Default::default()))
        .map_err(|e| rustler::Error::Term(Box::new(e.to_string())))?;
    Ok(atoms::ok())
}

#[rustler::nif(schedule = "DirtyCpu")]
fn generate_with_options(
    env: Env<'_>,
    model: ResourceArc<Model>,
    request: generate::GenerateRequest,
    options: rustler::Term<'_>,
    id: u64,
) -> NifResult<Atom> {
    let options = options::GenerateOptions::decode(options)?;
    model
        .jobs
        .try_send(Job::Generate(env.pid(), id, request, options))
        .map_err(|e| rustler::Error::Term(Box::new(e.to_string())))?;
    Ok(atoms::ok())
}

rustler::init!("Elixir.H3.Native");
