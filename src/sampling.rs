//! Explicit sampling recipes shared by command-line and embedding clients.
use crate::layout::Schedule;
use crate::{DenoiseParams, Result, Sampler};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Scheduler {
    #[default]
    Uniform,
    /// ComfyUI's discrete-flow 1,000-entry table and simple index selection.
    Simple,
}

impl Scheduler {
    pub(crate) fn schedule(self, points: usize, shift: f64) -> Schedule {
        if self == Self::Uniform {
            return Schedule::new(points, shift);
        }
        let evaluations = points.saturating_sub(1);
        let mut sigmas = Vec::with_capacity(points);
        let stride = 1000.0 / evaluations as f64;
        for x in 0..evaluations {
            let index = 999 - (x as f64 * stride) as usize;
            // Preserve each FP32 operation in ModelSamplingDiscreteFlow, including
            // the timestep multiplier round trip used to construct its table.
            let t = ((index + 1) as f32 / 1000.0) * 1000.0 / 1000.0;
            let shift = shift as f32;
            sigmas.push(shift * t / (1.0 + (shift - 1.0) * t));
        }
        sigmas.push(0.0);
        let timesteps = sigmas[..evaluations].iter().map(|s| 1.0 - s).collect();
        Schedule { sigmas, timesteps }
    }

    pub(crate) fn audio_schedule(
        self,
        points: usize,
        video_shift: f64,
        audio_shift: f64,
    ) -> Schedule {
        if self == Self::Uniform {
            return Schedule::new(points, audio_shift);
        }
        let mut s = self.schedule(points, video_shift);
        for sigma in &mut s.sigmas {
            let base = *sigma / (video_shift as f32 + *sigma * (1.0 - video_shift as f32));
            *sigma = audio_shift as f32 * base / (1.0 + (audio_shift as f32 - 1.0) * base);
        }
        s.timesteps = s.sigmas[..points - 1].iter().map(|s| 1.0 - s).collect();
        s
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SamplingPreset {
    FastH3,
}

/// Optional values override the preset. `steps` retains the legacy sigma-point meaning.
#[derive(Clone, Copy, Debug, Default)]
pub struct SamplingOptions {
    pub preset: Option<SamplingPreset>,
    pub steps: Option<usize>,
    pub evaluations: Option<usize>,
    pub sampler: Option<Sampler>,
    pub scheduler: Option<Scheduler>,
    pub video_shift: Option<f64>,
    pub audio_shift: Option<f64>,
}
impl SamplingOptions {
    pub fn resolve(
        self,
        mut params: DenoiseParams,
        explicit_checkpoint: bool,
    ) -> Result<DenoiseParams> {
        if self.steps.is_some() && self.evaluations.is_some() {
            return crate::error::invalid("steps and evaluations are mutually exclusive");
        }
        if self.preset == Some(SamplingPreset::FastH3) {
            if !explicit_checkpoint {
                return crate::error::invalid("FastH3 requires an explicit DiT checkpoint");
            }
            params.steps = 11;
            params.video_shift = 10.0;
            params.audio_shift = 3.0;
            params.sampler = Sampler::ResMultistep;
            params.scheduler = Scheduler::Simple;
        }
        if let Some(n) = self.evaluations {
            if !(1..=999).contains(&n) {
                return crate::error::invalid("evaluations must be 1..999");
            }
            params.steps = n + 1;
        }
        if let Some(n) = self.steps {
            params.steps = n;
        }
        if let Some(s) = self.sampler {
            params.sampler = s;
        }
        if let Some(s) = self.scheduler {
            params.scheduler = s;
        }
        if let Some(s) = self.video_shift {
            params.video_shift = s;
        }
        if let Some(s) = self.audio_shift {
            params.audio_shift = s;
        }
        validate(&params)?;
        Ok(params)
    }
}
pub fn validate(p: &DenoiseParams) -> Result<()> {
    if !(2..=1000).contains(&p.steps) {
        return crate::error::invalid("steps must be 2..1000");
    }
    for s in [p.video_shift, p.audio_shift] {
        if !s.is_finite() || s <= 0.0 || !(s as f32).is_finite() || s as f32 == 0.0 {
            return crate::error::invalid("sampling shifts must be finite and positive in FP32");
        }
    }
    for schedule in [
        p.scheduler.schedule(p.steps, p.video_shift),
        p.scheduler
            .audio_schedule(p.steps, p.video_shift, p.audio_shift),
    ] {
        if schedule
            .sigmas
            .iter()
            .any(|s| !s.is_finite() || !(0.0..=1.0).contains(s))
            || schedule.sigmas.windows(2).any(|w| w[0] <= w[1])
        {
            return crate::error::invalid(
                "sampling shifts produce a non-finite or non-decreasing FP32 sigma schedule",
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn preset_and_overrides() {
        let o = SamplingOptions {
            preset: Some(SamplingPreset::FastH3),
            ..Default::default()
        };
        assert!(o.resolve(DenoiseParams::default(), false).is_err());
        let p = o.resolve(DenoiseParams::default(), true).unwrap();
        assert_eq!((p.steps, p.video_shift, p.audio_shift), (11, 10.0, 3.0));
        assert_eq!(p.scheduler, Scheduler::Simple);
        let p = SamplingOptions {
            evaluations: Some(12),
            sampler: Some(Sampler::Euler),
            ..o
        }
        .resolve(p, true)
        .unwrap();
        assert_eq!((p.steps, p.sampler), (13, Sampler::Euler));
        let mut turbo = p;
        crate::adapter::TurboPreset::Eight.configure(&mut turbo);
        assert_eq!(
            (turbo.steps, turbo.scheduler, turbo.video_shift),
            (9, Scheduler::Uniform, 6.0)
        );
        assert!(SamplingOptions {
            steps: Some(12),
            evaluations: Some(12),
            ..o
        }
        .resolve(p, true)
        .is_err());
        for shift in [0.0, -1.0, f64::NAN, f64::INFINITY, 1e38, 1e-30] {
            assert!(SamplingOptions {
                video_shift: Some(shift),
                ..o
            }
            .resolve(p, true)
            .is_err());
        }
    }
    #[test]
    fn pinned_comfy_fp32_fixture() {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/fasth3_schedule.json")).unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let points = case["evaluations"].as_u64().unwrap() as usize + 1;
            let shift = case["video_shift"].as_f64().unwrap();
            let video = Scheduler::Simple.schedule(points, shift);
            let audio = Scheduler::Simple.audio_schedule(points, shift, 3.0);
            for (name, values) in [
                ("video", video.sigmas),
                ("audio", audio.sigmas),
                ("video_times", video.timesteps),
                ("audio_times", audio.timesteps),
            ] {
                let expected: Vec<u32> = case[name]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap() as u32)
                    .collect();
                assert_eq!(
                    values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    expected,
                    "{name} points={points} shift={shift}"
                );
            }
        }
    }
    #[test]
    fn table_selection() {
        for n in [1, 7, 8, 10, 12, 999] {
            let s = Scheduler::Simple.schedule(n + 1, 10.0);
            assert_eq!(s.sigmas.len(), n + 1);
            assert_eq!(s.sigmas[0], 1.0);
            assert_eq!(s.sigmas[n], 0.0);
            assert!(s.sigmas.windows(2).all(|w| w[0] > w[1]));
            assert_eq!(s.timesteps.len(), n);
        }
        assert_ne!(
            Scheduler::Simple.schedule(13, 10.0).sigmas,
            Schedule::new(13, 10.0).sigmas
        );
    }
}
