//! Decode qualification or CLI latent files with the normal native VAEs.
#[allow(dead_code)]
#[path = "../src/bin/h3/media.rs"]
mod media;
use anyhow::{bail, Context, Result};
use h3_hrx::{Config, ResidencyPolicy, Session, SessionOptions};
use std::path::Path;
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 6 {
        bail!("usage: render_latents PREFIX WIDTH HEIGHT FRAMES OUTPUT.mp4");
    }
    let shape = h3_hrx::shape_for(args[3].parse()?, args[2].parse()?, args[4].parse()?)
        .context("invalid dimensions")?;
    let read = |suffix: &str| -> Result<Vec<f32>> {
        let data = std::fs::read(format!("{}.{suffix}.f32", args[1]))?;
        if !data.len().is_multiple_of(4) {
            bail!("truncated float file");
        }
        Ok(data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| f32::from_le_bytes(*v))
            .collect())
    };
    let video = read("video")?;
    let audio = read("audio")?;
    // SAFETY: Hub-resolved VAE files remain immutable for this process.
    let mut session = unsafe {
        Session::new_with_options(
            Config::default(),
            SessionOptions {
                residency: ResidencyPolicy::StageScoped,
                ..Default::default()
            },
        )
    }?;
    let mut samples = vec![0f32; shape.audio_samples()];
    session.decode_audio(&audio, shape.audio_t as usize, &mut samples)?;
    if !samples.iter().all(|v| v.is_finite()) {
        bail!("non-finite decoded audio");
    }
    let wav = Path::new(&args[5]).with_extension("wav");
    std::fs::write(&wav, media::wav_bytes(&samples, (samples.len() / 2) as u32))?;
    let mut rgb = vec![0u8; shape.video_bytes()];
    session.decode_video(&shape, &video, &mut rgb)?;
    media::mux(
        Path::new(&args[5]),
        &wav,
        &rgb,
        shape.size().0,
        shape.size().1,
    )?;
    eprintln!(
        "decoded {} frames; audio peak {:.5}",
        shape.frames,
        samples.iter().map(|v| v.abs()).fold(0f32, f32::max)
    );
    Ok(())
}
