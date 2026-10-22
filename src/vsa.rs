//! Video sparse attention: live video tokens are grouped into 4×4×4 cubes.
use crate::{error::invalid, layout::Layout, Attention, Result};
pub(crate) mod device;

#[derive(Clone, Debug, Default, PartialEq)]
pub enum AttentionPattern {
    #[default]
    Dense,
    Vsa(VsaOptions),
}
#[derive(Clone, Debug, PartialEq)]
pub struct VsaOptions {
    pub keep_percent: f64,
    pub start: f64,
    pub end: f64,
    pub min_tokens: usize,
    /// Global zero-based DiT block indices that always run dense.
    pub dense_blocks: Vec<usize>,
}
impl Default for VsaOptions {
    fn default() -> Self {
        Self {
            keep_percent: 10.0,
            start: 0.2,
            end: 1.0,
            min_tokens: 12288,
            dense_blocks: Vec::new(),
        }
    }
}
impl VsaOptions {
    pub fn validate(&self) -> Result<()> {
        if !self.keep_percent.is_finite()
            || !(0.0..=100.0).contains(&self.keep_percent)
            || self.keep_percent == 0.0
        {
            return invalid("VSA keep_percent must be finite and in (0, 100]");
        }
        if !self.start.is_finite()
            || !self.end.is_finite()
            || self.start < 0.0
            || self.end > 1.0
            || self.start > self.end
        {
            return invalid("VSA requires 0 <= start <= end <= 1");
        }
        if self.min_tokens > 1 << 20 || self.dense_blocks.iter().any(|&b| b >= crate::model::BLOCKS)
        {
            return invalid("VSA token threshold or dense block index is out of range");
        }
        Ok(())
    }
    /// Comfy's Python round uses ties-to-even. Prefix blocks do not consume this budget.
    pub(crate) fn topk(&self, nonprefix: usize) -> usize {
        ((self.keep_percent / 100.0 * nonprefix as f64).round_ties_even() as usize)
            .max(1)
            .min(nonprefix.saturating_sub(1))
    }
    pub fn dense_reason(&self, tokens: usize, sigma: f32, shift: f64) -> Option<&'static str> {
        let threshold = |percent: f64| {
            let t = 1.0 - percent;
            (shift * t / (1.0 + (shift - 1.0) * t)) as f32
        };
        if tokens < self.min_tokens {
            Some("below_min_tokens")
        } else if sigma > threshold(self.start) || sigma < threshold(self.end) {
            Some("outside_sigma_window")
        } else {
            None
        }
    }
}
impl AttentionPattern {
    pub fn validate(&self, precision: Attention) -> Result<()> {
        if let Self::Vsa(options) = self {
            options.validate()?;
            if precision == Attention::I4 {
                return invalid("VSA supports f16 or i8 attention, not i4");
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Plan {
    pub src: Vec<i32>,
    pub inverse: Vec<i32>,
    pub lengths: Vec<i32>,
    pub prefix_blocks: usize,
}
impl Plan {
    pub fn new(layout: &Layout) -> Self {
        let mut src = Vec::new();
        let mut lengths = Vec::new();
        let mut segment = |start: usize, len: usize| {
            for row in (start..start + len).step_by(64) {
                let live = (start + len - row).min(64);
                lengths.push(live as i32);
                src.extend((row..row + live).map(|r| r as i32));
                src.resize(src.len() + 64 - live, -1);
            }
        };
        segment(0, layout.text_len);
        for s in &layout.ref_segs {
            segment(s.row0, s.rows);
        }
        segment(layout.text_len + layout.ref_rows, layout.audio_rows);
        let prefix_blocks = lengths.len();
        let (nt, nh, nw) = (layout.latent_t, layout.lat_h / 2, layout.lat_w / 2);
        let video0 = layout.seq_len - layout.video_rows;
        for t in (0..nt).step_by(4) {
            for h in (0..nh).step_by(4) {
                for w in (0..nw).step_by(4) {
                    let start = src.len();
                    for tt in t..(t + 4).min(nt) {
                        for hh in h..(h + 4).min(nh) {
                            for ww in w..(w + 4).min(nw) {
                                src.push((video0 + (tt * nh + hh) * nw + ww) as i32);
                            }
                        }
                    }
                    lengths.push((src.len() - start) as i32);
                    src.resize(start + 64, -1);
                }
            }
        }
        let mut inverse = vec![-1; layout.seq_len];
        for (i, &row) in src.iter().enumerate() {
            if row >= 0 {
                inverse[row as usize] = i as i32;
            }
        }
        assert!(
            inverse.iter().all(|&i| i >= 0),
            "VSA layout must cover every live token"
        );
        Self {
            src,
            inverse,
            lengths,
            prefix_blocks,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cube_layout_is_bijective_and_pads_each_prefix_segment() {
        let layout = Layout::new(65, 5, 10, 14, 3, &[], &[]).unwrap();
        let p = Plan::new(&layout);
        assert_eq!(p.prefix_blocks, 3);
        assert_eq!(&p.lengths[..3], &[64, 1, 6]);
        for (row, &i) in p.inverse.iter().enumerate() {
            assert_eq!(p.src[i as usize], row as i32);
        }
        for (tile, &len) in p.src.chunks(64).zip(&p.lengths) {
            assert!(tile[..len as usize].iter().all(|&r| r >= 0));
            assert!(tile[len as usize..].iter().all(|&r| r == -1));
        }
    }
    #[test]
    fn references_and_keyframes_are_separate_exact_prefix_segments() {
        let refs = [
            crate::layout::Ref {
                kind: 0,
                latent_t: 1,
                lat_h: 6,
                lat_w: 10,
                audio_t: 0,
                has_audio: false,
            },
            crate::layout::Ref {
                kind: 1,
                latent_t: 0,
                lat_h: 0,
                lat_w: 0,
                audio_t: 7,
                has_audio: true,
            },
            crate::layout::Ref {
                kind: 2,
                latent_t: 3,
                lat_h: 8,
                lat_w: 10,
                audio_t: 9,
                has_audio: true,
            },
        ];
        let keys = [crate::layout::Keyframe {
            frame_index: 0,
            audio_t: 4,
            has_audio: true,
        }];
        let layout = Layout::new(11, 5, 10, 14, 6, &refs, &keys).unwrap();
        let plan = Plan::new(&layout);
        for (row, &i) in plan.inverse.iter().enumerate() {
            assert_eq!(plan.src[i as usize], row as i32);
        }
        let prefix_rows = layout.seq_len - layout.video_rows;
        assert!(plan.src[..plan.prefix_blocks * 64]
            .iter()
            .filter(|&&r| r >= 0)
            .all(|&r| (r as usize) < prefix_rows));
        assert_eq!(
            plan.lengths[..plan.prefix_blocks].iter().sum::<i32>() as usize,
            prefix_rows
        );
    }
    #[test]
    fn routing_budget_and_sigma_window() {
        let o = VsaOptions::default();
        assert_eq!(o.topk(25), 2);
        assert_eq!(o.topk(35), 4);
        assert_eq!(o.topk(1), 0);
        assert_eq!(
            o.dense_reason(12288, 1.0, 10.0),
            Some("outside_sigma_window")
        );
        assert_eq!(o.dense_reason(12288, 0.9, 10.0), None);
        assert_eq!(o.dense_reason(12287, 0.5, 10.0), Some("below_min_tokens"));
    }
}
