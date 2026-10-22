//! Reproducible FastH3 matrix and lifecycle qualification on native hardware.
//! Checkpoint files must remain immutable for the lifetime of this process.
use h3_hrx::{
    AttentionPattern, Config, Control, DenoiseParams, Noise, Sampler, SamplingOptions,
    SamplingPreset, Session, SessionOptions, VsaOptions,
};
use std::{path::PathBuf, time::Instant};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 {
        return Err("usage: qualify_fasth3 CHECKPOINT OUTPUT_DIR dense|vsa|observe".into());
    }
    let dir = PathBuf::from(&args[2]);
    std::fs::create_dir_all(&dir)?;
    let pattern = match args[3].as_str() {
        "dense" => AttentionPattern::Dense,
        "vsa" | "observe" => AttentionPattern::Vsa(VsaOptions {
            min_tokens: 0,
            ..Default::default()
        }),
        _ => return Err("unknown attention pattern".into()),
    };
    let config = Config {
        dit: Some(args[1].clone().into()),
        attention_pattern: pattern,
        ..Default::default()
    };
    let manager = hrx::residency::ResidencyManager::new(64 << 30)?;
    let context = hrx::inference::ModelContext::new(hrx::execution::RuntimeOptions {
        memory_budget: Some(manager.budget()),
        ..Default::default()
    })?;
    // SAFETY: the caller supplies an immutable checkpoint; Hub files are immutable.
    let mut session = unsafe {
        Session::new_in(
            config,
            SessionOptions {
                residency: h3_hrx::ResidencyPolicy::StageScoped,
                cache: if args[3] == "observe" {
                    h3_hrx::CachePolicy::Observe
                } else {
                    h3_hrx::CachePolicy::Off
                },
                ..Default::default()
            },
            &context,
        )
    }?;
    let ids = h3_hrx::Tokenizer::new()?.encode("A red ball rolls across a wooden table.")?;
    let base = SamplingOptions {
        preset: Some(SamplingPreset::FastH3),
        ..Default::default()
    }
    .resolve(
        DenoiseParams {
            width: 256,
            height: 256,
            frames: 5,
            seed: 7,
            ..Default::default()
        },
        true,
    )?;
    let mut last = None;
    for sampler in [Sampler::ResMultistep, Sampler::Euler] {
        for evaluations in [8, 10, 12] {
            let p = SamplingOptions {
                evaluations: Some(evaluations),
                sampler: Some(sampler),
                ..Default::default()
            }
            .resolve(base, true)?;
            let start = Instant::now();
            let out = session.denoise(&ids, &p, Noise::default(), &[], &[], None)?;
            assert!(out.video.iter().chain(&out.audio).all(|x| x.is_finite()));
            let name = format!("{sampler:?}-{evaluations}");
            std::fs::write(
                dir.join(format!("{name}.video.f32")),
                bytemuck::cast_slice(&out.video),
            )?;
            std::fs::write(
                dir.join(format!("{name}.audio.f32")),
                bytemuck::cast_slice(&out.audio),
            )?;
            println!(
                "{}",
                serde_json::json!({"case":name,"seconds":start.elapsed().as_secs_f64(),"reserved_bytes":manager.statistics().reserved_bytes})
            );
            last = Some((p, out));
        }
    }
    let (p, expected) = last.unwrap();
    let error = session.denoise(
        &ids,
        &p,
        Noise::default(),
        &[],
        &[],
        Some(&mut |step, _, _| {
            if step >= 4 {
                Control::Cancel
            } else {
                Control::Continue
            }
        }),
    );
    assert!(matches!(error, Err(h3_hrx::Error::Cancelled)));
    let changed = DenoiseParams {
        width: 128,
        height: 128,
        steps: 3,
        ..p
    };
    session.denoise(&ids, &changed, Noise::default(), &[], &[], None)?;
    let replay = session.denoise(&ids, &p, Noise::default(), &[], &[], None)?;
    assert_eq!(
        expected.video, replay.video,
        "video after cancellation/layout change"
    );
    assert_eq!(
        expected.audio, replay.audio,
        "audio after cancellation/layout change"
    );
    drop(session);
    assert_eq!(
        manager.statistics().reserved_bytes,
        0,
        "model drop must retire budget leases"
    );
    println!("cancellation, layout change, repeatability and budget retirement: passed");
    Ok(())
}
