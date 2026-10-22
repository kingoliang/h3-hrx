//! Hardware qualification reuses prepared text without retaining the large encoder.
use super::*;
#[test]
#[ignore = "requires explicit immutable H3_QUALIFY_CHECKPOINT and H3_QUALIFY_OUT"]
fn fasth3_matrix() -> Result<()> {
    let path = std::env::var("H3_QUALIFY_CHECKPOINT").expect("checkpoint path");
    let dir = std::path::PathBuf::from(std::env::var("H3_QUALIFY_OUT").expect("output directory"));
    std::fs::create_dir_all(&dir).unwrap();
    let mode = std::env::var("H3_QUALIFY_PATTERN").unwrap_or_else(|_| "vsa".into());
    let full = std::env::var_os("H3_QUALIFY_FULL").is_some();
    let manager = hrx::residency::ResidencyManager::new(48 << 30)?;
    let budget = manager.budget();
    let mut stream = hrx::Stream::open()?.with_memory_budget(budget.clone());
    let c = Compiler::new(None, "");
    let mut prof = Profile::default();
    let mut dit = unsafe { Dit::open(&mut stream, &path) }?;
    if mode != "dense" {
        dit.set_attention_pattern(crate::AttentionPattern::Vsa(crate::VsaOptions {
            min_tokens: if full { 12288 } else { 0 },
            ..Default::default()
        }));
    }
    let ids = crate::Tokenizer::new()?.encode("A red ball rolls across a wooden table.")?;
    let base = crate::SamplingOptions {
        preset: Some(crate::SamplingPreset::FastH3),
        ..Default::default()
    }
    .resolve(
        DenoiseParams {
            width: if full { 1344 } else { 256 },
            height: if full { 768 } else { 256 },
            frames: if full { 124 } else { 5 },
            seed: 7,
            ..Default::default()
        },
        true,
    )?;
    let te_path = crate::models::Resolver::new().find(crate::models::TE)?;
    let mut te = unsafe { TextEncoder::open(&mut stream, &te_path) }?;
    dit.prepare_denoise(
        &mut stream,
        &c,
        &mut prof,
        &mut te,
        &ids,
        &base,
        &[],
        &[],
        None,
        None,
    )?;
    let mut conditioning = vec![0f32; ids.len() * HID];
    dit.read_rows(&mut stream, ids.len(), &mut conditioning)?;
    stream.synchronize()?;
    drop(te);
    let cache = if mode == "observe" {
        crate::CachePolicy::Observe
    } else {
        crate::CachePolicy::Off
    };
    let mut run = |p: DenoiseParams, cancel: bool| -> Result<Latents> {
        let sh = crate::shape_for(p.height, p.width, p.frames).unwrap();
        let lay = crate::layout::Layout::new(
            ids.len(),
            sh.latent_t as usize,
            sh.lat_h as usize,
            sh.lat_w as usize,
            sh.audio_t as usize,
            &[],
            &[],
        )
        .unwrap();
        dit.ensure_seq(&mut stream, lay.seq_len)?;
        let seq = dit.seq.as_mut().unwrap();
        let (cos, sin) = lay.rotary_tables(&dit.cond.as_ref().unwrap().inv_freq);
        seq.cls.write(&mut stream, &lay.adaln_rows, CLASSES)?;
        seq.tcls.write(&mut stream, &lay.tclass[ids.len()..], 2)?;
        stream.upload(
            seq.x.slice(0, conditioning.len() * 4),
            bytemuck::cast_slice(&conditioning),
        )?;
        stream.upload_at(&seq.cos, 0, bytemuck::cast_slice(&cos))?;
        stream.upload_at(&seq.sin, 0, bytemuck::cast_slice(&sin))?;
        dit.sample_prepared(
            &mut stream,
            &c,
            &mut prof,
            lay,
            &p,
            8,
            Noise::default(),
            &[],
            &[],
            Some(&mut |step, total, seconds| {
                eprintln!(
                    "step {step}/{total}: {seconds:.3}s; resident {} MiB",
                    budget.reserved_bytes() >> 20
                );
                if cancel {
                    Control::Cancel
                } else {
                    Control::Continue
                }
            }),
            cache,
            None,
        )
    };
    let cases = if full {
        vec![(Sampler::ResMultistep, 2), (Sampler::ResMultistep, 10)]
    } else {
        vec![
            (Sampler::ResMultistep, 8),
            (Sampler::ResMultistep, 10),
            (Sampler::ResMultistep, 12),
            (Sampler::Euler, 8),
            (Sampler::Euler, 10),
            (Sampler::Euler, 12),
        ]
    };
    let mut last = None;
    for (sampler, evaluations) in cases {
        let p = DenoiseParams {
            steps: evaluations + 1,
            sampler,
            ..base
        };
        let start = std::time::Instant::now();
        let out = run(p, false)?;
        assert!(out.video.iter().chain(&out.audio).all(|x| x.is_finite()));
        let name = format!("{sampler:?}-{evaluations}");
        std::fs::write(
            dir.join(format!("{name}.video.f32")),
            bytemuck::cast_slice(&out.video),
        )
        .unwrap();
        std::fs::write(
            dir.join(format!("{name}.audio.f32")),
            bytemuck::cast_slice(&out.audio),
        )
        .unwrap();
        eprintln!(
            "{}",
            serde_json::json!({"case":name,"seconds":start.elapsed().as_secs_f64(),"reserved_bytes":budget.reserved_bytes()})
        );
        last = Some((p, out));
    }
    if !full {
        let (p, expected) = last.unwrap();
        assert!(matches!(run(p, true), Err(crate::Error::Cancelled)));
        run(
            DenoiseParams {
                width: 128,
                height: 128,
                steps: 3,
                ..p
            },
            false,
        )?;
        let replay = run(p, false)?;
        assert_eq!(expected.video, replay.video);
        assert_eq!(expected.audio, replay.audio);
    }
    stream.synchronize()?;
    drop(dit);
    stream.synchronize()?;
    drop(stream);
    assert_eq!(budget.reserved_bytes(), 0);
    Ok(())
}
