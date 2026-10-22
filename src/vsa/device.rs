//! GPU scratch is shared across blocks; previous-step centering is owned per global block.
use super::{Plan, VsaOptions};
use crate::compile::{num, Compiler, Kernel, Result};
use crate::dispatch::{emit, Profile, Sink};
use hrx::{Buffer, View};

pub(crate) struct Runtime {
    pub plan: Plan,
    _host_reservation: Option<hrx::residency::MemoryReservation>,
    width: usize,
    heads: usize,
    gather: Kernel,
    scatter: Kernel,
    pool: Kernel,
    scores: Kernel,
    route: Kernel,
    coarse: Kernel,
    gated_scatter: Kernel,
    vm: Buffer,
    coarse_buf: Buffer,
    attention: Kernel,
    src: Buffer,
    inverse: Buffer,
    lengths: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    qm: Buffer,
    km: Buffer,
    scores_buf: Buffer,
    routes: Buffer,
    out: Buffer,
    integer: Option<Integer>,
}
struct Integer {
    prep_q: Kernel,
    prep_k: Kernel,
    transpose: Kernel,
    update_mean: Kernel,
    qi: Buffer,
    ki: Buffer,
    qs: Buffer,
    ks: Buffer,
    vt: Buffer,
    zero: Buffer,
    means: Vec<Buffer>,
}
impl Runtime {
    pub fn new(
        c: &Compiler,
        stream: &mut hrx::Stream,
        plan: Plan,
        options: &VsaOptions,
        heads: usize,
        bits: usize,
        layers: usize,
    ) -> Result<Self> {
        let host_bytes = (plan.src.len() + plan.inverse.len() + plan.lengths.len()) * 4;
        let host_reservation = stream
            .memory_budget()
            .map(|b| b.reserve(host_bytes))
            .transpose()?;
        let width = heads * 128;
        let tokens = plan.inverse.len();
        let n = plan.src.len();
        let blocks = plan.lengths.len();
        let cfg = [
            ("tokens", tokens),
            ("padded", n),
            ("width", width),
            ("blocks", blocks),
            ("heads", heads),
            ("prefix", plan.prefix_blocks),
            ("topk", options.topk(blocks - plan.prefix_blocks)),
        ]
        .into_iter()
        .map(|(k, v)| (format!("h3.vsa.{k}"), v.to_string()))
        .collect();
        let gather = c.get(stream, "vsa_ops", "h3_vsa_gather", &cfg)?;
        let scatter = c.get(stream, "vsa_ops", "h3_vsa_scatter", &cfg)?;
        let pool = c.get(stream, "vsa_ops", "h3_vsa_pool", &cfg)?;
        let scores = c.get(stream, "vsa_ops", "h3_vsa_scores", &cfg)?;
        let route = c.get(stream, "vsa_ops", "h3_vsa_route", &cfg)?;
        let coarse = c.get(stream, "vsa_ops", "h3_vsa_coarse", &cfg)?;
        let gated_scatter = c.get(stream, "vsa_ops", "h3_vsa_gated_scatter", &cfg)?;
        let stem = format!("vsa_attention_{bits}");
        let mut acfg: Vec<_> = [
            ("tokens", n),
            ("token_capacity", n),
            ("q_stride", width),
            ("kv_stride", width),
            ("out_stride", width),
        ]
        .into_iter()
        .map(|(k, v)| (format!("h3.{stem}.{k}"), v.to_string()))
        .collect();
        acfg.push((format!("h3.{stem}.scale"), num(1.0 / 128.0f64.sqrt())));
        let attention = c.get(stream, &stem, &format!("h3_{stem}"), &acfg)?;
        let upload = |stream: &mut hrx::Stream, values: &[i32]| -> Result<Buffer> {
            let b = stream.allocate(values.len() * 4)?;
            stream.upload(b.binding(), bytemuck::cast_slice(values))?;
            Ok(b)
        };
        let src = upload(stream, &plan.src)?;
        let inverse = upload(stream, &plan.inverse)?;
        let lengths = upload(stream, &plan.lengths)?;
        let integer = if bits == 8 {
            let mut cfg = vec![
                ("h3.prepare_qk_i8.row_stride".into(), width.to_string()),
                ("h3.prepare_qk_i8.head_offset".into(), "0".into()),
                ("h3.prepare_qk_i8.heads".into(), heads.to_string()),
                (
                    "h3.prepare_qk_i8.extra_scale".into(),
                    num(1.0 / 128.0 / 128.0f64.sqrt()),
                ),
            ];
            let prep_q = c.get(stream, "prepare_qk_i8", "h3_prepare_qk_i8", &cfg)?;
            cfg[3].1 = "1".into();
            let prep_k = c.get(stream, "prepare_qk_i8", "h3_prepare_qk_i8", &cfg)?;
            let transpose = c.get(
                stream,
                "transpose_f16",
                "h3_transpose_f16",
                &vec![
                    ("h3.transpose_f16.width".into(), width.to_string()),
                    ("h3.transpose_f16.row_capacity".into(), n.to_string()),
                ],
            )?;
            let update_mean = c.get(
                stream,
                "vsa_kmean",
                "h3_vsa_kmean",
                &vec![("h3.vsa_kmean.width".into(), width.to_string())],
            )?;
            let zero = stream.allocate(width * 4)?;
            stream.fill(zero.binding(), 0)?;
            let means = (0..layers)
                .map(|_| {
                    let b = stream.allocate(width * 4)?;
                    stream.fill(b.binding(), 0)?;
                    Ok(b)
                })
                .collect::<Result<Vec<_>>>()?;
            Some(Integer {
                prep_q,
                prep_k,
                transpose,
                update_mean,
                zero,
                means,
                qi: stream.allocate(n * width)?,
                ki: stream.allocate(n * width)?,
                qs: stream.allocate(n * heads * 4)?,
                ks: stream.allocate(n * heads * 4)?,
                vt: stream.allocate(n * width * 2)?,
            })
        } else {
            None
        };
        c.flush(stream)?;
        Ok(Self {
            plan,
            _host_reservation: host_reservation,
            width,
            heads,
            gather,
            scatter,
            pool,
            scores,
            route,
            attention,
            src,
            inverse,
            lengths,
            coarse,
            gated_scatter,
            vm: stream.allocate(blocks * width * 4)?,
            coarse_buf: stream.allocate(blocks * width * 4)?,
            q: stream.allocate(n * width * 2)?,
            k: stream.allocate(n * width * 2)?,
            v: stream.allocate(n * width * 2)?,
            qm: stream.allocate(blocks * width * 4)?,
            km: stream.allocate(blocks * width * 4)?,
            scores_buf: stream.allocate(heads * blocks * blocks * 4)?,
            routes: stream.allocate(heads * blocks * blocks * 4)?,
            out: stream.allocate(n * width * 2)?,
            integer,
        })
    }
    pub fn reset(&self, stream: &mut hrx::Stream) -> Result<()> {
        if let Some(i) = &self.integer {
            for mean in &i.means {
                stream.fill(mean.binding(), 0)?;
            }
        }
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    pub fn emit<'g>(
        &'g self,
        sink: &mut Sink<'_, 'g>,
        prof: &mut Profile,
        block: usize,
        q: View<'g>,
        k: View<'g>,
        v: View<'g>,
        out: View<'g>,
        gate: Option<View<'g>>,
    ) -> Result<()> {
        let n = self.plan.src.len();
        let tokens = self.plan.inverse.len();
        let b = self.plan.lengths.len();
        let w = self.width;
        let n32 = n as u32;
        for (input, output) in [(q, &self.q), (k, &self.k), (v, &self.v)] {
            emit(
                sink,
                &self.gather,
                Some(prof),
                "VSA gather",
                &[1],
                &[1],
                &[self.src.binding(), input, output.binding()],
                &[n * 4, tokens * w * 2, n * w * 2],
            )?;
        }
        for (input, output) in [(&self.q, &self.qm), (&self.k, &self.km)] {
            emit(
                sink,
                &self.pool,
                Some(prof),
                "VSA pool",
                &[1],
                &[1],
                &[self.lengths.binding(), input.binding(), output.binding()],
                &[b * 4, n * w * 2, b * w * 4],
            )?;
        }
        emit(
            sink,
            &self.scores,
            Some(prof),
            "VSA block scores",
            &[1],
            &[1],
            &[
                self.qm.binding(),
                self.km.binding(),
                self.scores_buf.binding(),
            ],
            &[b * w * 4, b * w * 4, self.heads * b * b * 4],
        )?;
        emit(
            sink,
            &self.route,
            Some(prof),
            "VSA route",
            &[1],
            &[1],
            &[self.scores_buf.binding(), self.routes.binding()],
            &[self.heads * b * b * 4, self.heads * b * b * 4],
        )?;
        #[cfg(test)]
        if block == 0 && std::env::var_os("H3_VSA_DIAGNOSTICS").is_some() {
            if let Sink::Stream(stream) = sink {
                for (name, buf) in [("q_means", &self.qm), ("k_means", &self.km)] {
                    let mut values = vec![0f32; b * w];
                    stream.read_blocking(buf.binding(), bytemuck::cast_slice_mut(&mut values))?;
                    assert!(values.iter().all(|v| v.is_finite()), "non-finite {name}");
                    eprintln!(
                        "VSA {name}: nonfinite={} nonzero={} min={} max={}",
                        values.iter().filter(|v| !v.is_finite()).count(),
                        values.iter().filter(|&&v| v != 0.0).count(),
                        values.iter().copied().fold(f32::INFINITY, f32::min),
                        values.iter().copied().fold(f32::NEG_INFINITY, f32::max)
                    );
                }
                let mut values = vec![0i32; self.heads * b * b];
                stream
                    .read_blocking(self.routes.binding(), bytemuck::cast_slice_mut(&mut values))?;
                let counts: Vec<_> = values
                    .chunks_exact(b)
                    .map(|row| row.iter().sum::<i32>())
                    .collect();
                eprintln!(
                    "VSA routes: blocks={b} min={} max={} average={}",
                    counts.iter().min().unwrap(),
                    counts.iter().max().unwrap(),
                    counts.iter().map(|&n| n as f64).sum::<f64>() / counts.len() as f64
                );
            }
        }
        if let Some(i) = &self.integer {
            for (kernel, input, mean, codes, scales) in [
                (&i.prep_q, &self.q, &i.zero, &i.qi, &i.qs),
                (&i.prep_k, &self.k, &i.means[block], &i.ki, &i.ks),
            ] {
                emit(
                    sink,
                    kernel,
                    Some(prof),
                    "VSA integer operands",
                    &[n32],
                    &[n32],
                    &[
                        input.binding(),
                        mean.binding(),
                        codes.binding(),
                        scales.binding(),
                    ],
                    &[n * w * 2, w * 4, n * w, n * self.heads * 4],
                )?;
            }
            emit(
                sink,
                &i.transpose,
                Some(prof),
                "VSA transpose",
                &[n32],
                &[n32],
                &[self.v.binding(), i.vt.binding()],
                &[n * w * 2, n * w * 2],
            )?;
            emit(
                sink,
                &self.attention,
                Some(prof),
                "VSA attention",
                &[n32, self.heads as u32],
                &[n32],
                &[
                    i.qi.binding(),
                    i.qs.binding(),
                    i.ki.binding(),
                    i.ks.binding(),
                    i.vt.binding(),
                    self.out.binding(),
                    self.routes.binding(),
                    self.lengths.binding(),
                ],
                &[
                    n * w,
                    n * self.heads * 4,
                    n * w,
                    n * self.heads * 4,
                    n * w * 2,
                    n * w * 2,
                    self.heads * b * b * 4,
                    b * 4,
                ],
            )?;
            emit(
                sink,
                &i.update_mean,
                Some(prof),
                "VSA statistics",
                &[b as u32],
                &[b as u32],
                &[self.km.binding(), i.means[block].binding()],
                &[b * w * 4, w * 4],
            )?;
        } else {
            emit(
                sink,
                &self.attention,
                Some(prof),
                "VSA attention",
                &[n32, self.heads as u32],
                &[n32],
                &[
                    self.q.binding(),
                    self.k.binding(),
                    self.v.binding(),
                    self.out.binding(),
                    self.routes.binding(),
                    self.lengths.binding(),
                ],
                &[
                    n * w * 2,
                    n * w * 2,
                    n * w * 2,
                    n * w * 2,
                    self.heads * b * b * 4,
                    b * 4,
                ],
            )?;
        }
        if let Some(gate) = gate {
            emit(
                sink,
                &self.pool,
                Some(prof),
                "VSA value means",
                &[1],
                &[1],
                &[self.lengths.binding(), self.v.binding(), self.vm.binding()],
                &[b * 4, n * w * 2, b * w * 4],
            )?;
            emit(
                sink,
                &self.coarse,
                Some(prof),
                "VSA coarse attention",
                &[1],
                &[1],
                &[
                    self.scores_buf.binding(),
                    self.vm.binding(),
                    self.coarse_buf.binding(),
                ],
                &[self.heads * b * b * 4, b * w * 4, b * w * 4],
            )?;
            return emit(
                sink,
                &self.gated_scatter,
                Some(prof),
                "VSA gated scatter",
                &[1],
                &[1],
                &[
                    self.inverse.binding(),
                    self.out.binding(),
                    gate,
                    self.coarse_buf.binding(),
                    out,
                ],
                &[
                    tokens * 4,
                    n * w * 2,
                    tokens * w * 2,
                    b * w * 4,
                    tokens * w * 2,
                ],
            );
        }
        emit(
            sink,
            &self.scatter,
            Some(prof),
            "VSA scatter",
            &[1],
            &[1],
            &[self.inverse.binding(), self.out.binding(), out],
            &[tokens * 4, n * w * 2, tokens * w * 2],
        )
    }
}

#[cfg(all(test, feature = "gpu-tests"))]
mod tests {
    use super::*;
    #[test]
    #[ignore = "full-size GPU performance diagnostic; use H3_PROFILE=device"]
    fn full_size_profile() {
        let mut stream = hrx::Stream::open().unwrap();
        let c = Compiler::new(None, "");
        let layout = crate::layout::Layout::new(9, 37, 48, 84, 207, &[], &[]).unwrap();
        let plan = Plan::new(&layout);
        let heads = 56;
        let count = plan.inverse.len() * heads * 128;
        let data: Vec<_> = (0..count)
            .map(|i| {
                let mut x = (i as u32).wrapping_mul(747796405).wrapping_add(2891336453);
                x = ((x >> ((x >> 28) + 4)) ^ x).wrapping_mul(277803737);
                x = (x >> 22) ^ x;
                half::f16::from_f32((x as f64 / u32::MAX as f64 * 2.0 - 1.0) as f32)
            })
            .collect();
        let input = stream.allocate(count * 2).unwrap();
        stream
            .upload(input.binding(), bytemuck::cast_slice(&data))
            .unwrap();
        drop(data);
        let out = stream.allocate(count * 2).unwrap();
        let runtime =
            Runtime::new(&c, &mut stream, plan, &VsaOptions::default(), heads, 8, 1).unwrap();
        let mut profile = Profile::from_env();
        runtime
            .emit(
                &mut Sink::Stream(&mut stream),
                &mut profile,
                0,
                input.binding(),
                input.binding(),
                input.binding(),
                out.binding(),
                None,
            )
            .unwrap();
        stream.synchronize().unwrap();
        eprintln!("{}", profile.report(0.0));
    }
    #[test]
    fn sparse_attention_matches_independent_masked_oracle() {
        let mut stream = hrx::Stream::open().unwrap();
        let c = Compiler::new(None, "");
        let heads = 2;
        let width = heads * 128;
        let lengths = vec![3, 64, 49, 64, 8, 22];
        let mut src = Vec::new();
        let mut inverse = Vec::new();
        for &len in &lengths {
            for _ in 0..len {
                inverse.push(src.len() as i32);
                src.push((inverse.len() - 1) as i32);
            }
            src.resize(src.len() + 64 - len as usize, -1);
        }
        let plan = Plan {
            src,
            inverse,
            lengths,
            prefix_blocks: 1,
        };
        let t = plan.inverse.len();
        let data: Vec<Vec<half::f16>> = (0..3)
            .map(|which| {
                (0..t * width)
                    .map(|i| {
                        half::f16::from_f32(
                            ((i * 37 + which * 113) as f32 * 0.017).sin() * 0.8
                                + ((i / width + which * 3) as f32 * 0.19).cos() * 0.3,
                        )
                    })
                    .collect()
            })
            .collect();
        let options = VsaOptions {
            keep_percent: 30.0,
            ..Default::default()
        };
        let mut expected = vec![0f32; t * width];
        let nb = plan.lengths.len();
        let means: Vec<Vec<f32>> = data
            .iter()
            .map(|x| {
                let mut m = vec![0f32; nb * width];
                for (b, &len) in plan.lengths.iter().enumerate() {
                    for j in 0..len as usize {
                        let row = plan.src[b * 64 + j] as usize;
                        for d in 0..width {
                            m[b * width + d] += x[row * width + d].to_f32() / len as f32;
                        }
                    }
                }
                m
            })
            .collect();
        for h in 0..heads {
            for qb in 0..nb {
                let scores: Vec<f32> = (0..nb)
                    .map(|kb| {
                        (0..128)
                            .map(|d| {
                                means[0][qb * width + h * 128 + d]
                                    * means[1][kb * width + h * 128 + d]
                            })
                            .sum()
                    })
                    .collect();
                let mut ranked = scores[1..].to_vec();
                ranked.sort_by(|a, b| b.total_cmp(a));
                let threshold = ranked[options.topk(nb - 1) - 1];
                let keys: Vec<usize> = (0..nb)
                    .filter(|&kb| {
                        qb == 0 || kb == 0 || qb.abs_diff(kb) <= 1 || scores[kb] >= threshold
                    })
                    .flat_map(|kb| (0..plan.lengths[kb] as usize).map(move |j| kb * 64 + j))
                    .map(|i| plan.src[i] as usize)
                    .collect();
                for qi in 0..plan.lengths[qb] as usize {
                    let row = plan.src[qb * 64 + qi] as usize;
                    let logits: Vec<f32> = keys
                        .iter()
                        .map(|&key| {
                            (0..128)
                                .map(|d| {
                                    data[0][row * width + h * 128 + d].to_f32()
                                        * data[1][key * width + h * 128 + d].to_f32()
                                })
                                .sum::<f32>()
                                / 128f32.sqrt()
                        })
                        .collect();
                    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let probs: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
                    let sum: f32 = probs.iter().sum();
                    for d in 0..128 {
                        expected[row * width + h * 128 + d] = keys
                            .iter()
                            .zip(&probs)
                            .map(|(&key, p)| p / sum * data[2][key * width + h * 128 + d].to_f32())
                            .sum();
                    }
                }
            }
        }
        let gates: Vec<_> = (0..t * width)
            .map(|i| half::f16::from_f32(if i % 3 == 0 { -1.25 } else { 0.75 }))
            .collect();
        let mut gated_expected = expected.clone();
        for h in 0..heads {
            for qb in 0..nb {
                let logits: Vec<f32> = (0..nb)
                    .map(|kb| {
                        (0..128)
                            .map(|d| {
                                means[0][qb * width + h * 128 + d]
                                    * means[1][kb * width + h * 128 + d]
                            })
                            .sum::<f32>()
                            / 128f32.sqrt()
                    })
                    .collect();
                let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let probs: Vec<f32> = logits.iter().map(|v| (v - max).exp()).collect();
                let sum: f32 = probs.iter().sum();
                for d in 0..128 {
                    let coarse: f32 = (0..nb)
                        .map(|kb| probs[kb] / sum * means[2][kb * width + h * 128 + d])
                        .sum();
                    for qi in 0..plan.lengths[qb] as usize {
                        let i = plan.src[qb * 64 + qi] as usize * width + h * 128 + d;
                        gated_expected[i] += gates[i].to_f32() * coarse;
                    }
                }
            }
        }
        let dump = std::env::var_os("H3_VSA_FIXTURE").map(std::path::PathBuf::from);
        if let Some(dir) = &dump {
            std::fs::create_dir_all(dir).unwrap();
            for (name, values) in [
                ("q", &data[0]),
                ("k", &data[1]),
                ("v", &data[2]),
                ("gate", &gates),
            ] {
                std::fs::write(
                    dir.join(format!("{name}.f16")),
                    bytemuck::cast_slice(values),
                )
                .unwrap();
            }
            std::fs::write(dir.join("layout.json"),serde_json::to_vec(&serde_json::json!({"src":plan.src,"inverse":plan.inverse,"lengths":plan.lengths,"prefix":plan.prefix_blocks,"heads":heads,"keep_percent":options.keep_percent})).unwrap()).unwrap();
        }
        let gate_buffer = stream.allocate(t * width * 2).unwrap();
        stream
            .upload(gate_buffer.binding(), bytemuck::cast_slice(&gates))
            .unwrap();
        let buffers: Vec<_> = data
            .iter()
            .map(|x| {
                let b = stream.allocate(x.len() * 2).unwrap();
                stream.upload(b.binding(), bytemuck::cast_slice(x)).unwrap();
                b
            })
            .collect();
        let out = stream.allocate(t * width * 2).unwrap();
        for bits in [16, 8] {
            let runtime =
                Runtime::new(&c, &mut stream, plan.clone(), &options, heads, bits, 1).unwrap();
            let mut profile = Profile::default();
            runtime
                .emit(
                    &mut Sink::Stream(&mut stream),
                    &mut profile,
                    0,
                    buffers[0].binding(),
                    buffers[1].binding(),
                    buffers[2].binding(),
                    out.binding(),
                    None,
                )
                .unwrap();
            let mut actual = vec![half::f16::ZERO; t * width];
            stream
                .read_blocking(out.binding(), bytemuck::cast_slice_mut(&mut actual))
                .unwrap();
            let error = (actual
                .iter()
                .zip(&expected)
                .map(|(a, e)| (a.to_f32() - e).powi(2))
                .sum::<f32>()
                / expected.iter().map(|e| e * e).sum::<f32>())
            .sqrt();
            if let Some(dir) = &dump {
                std::fs::write(
                    dir.join(format!("out-{bits}.f16")),
                    bytemuck::cast_slice(&actual),
                )
                .unwrap();
            }
            eprintln!("VSA {bits}: normalized RMSE={error}");
            assert!(error < if bits == 16 { 0.01 } else { 0.03 });
            runtime.reset(&mut stream).unwrap();
            let mut graph = stream.graph().unwrap();
            runtime
                .emit(
                    &mut Sink::Graph {
                        graph: &mut graph,
                        after: Default::default(),
                    },
                    &mut profile,
                    0,
                    buffers[0].binding(),
                    buffers[1].binding(),
                    buffers[2].binding(),
                    out.binding(),
                    None,
                )
                .unwrap();
            let mut exec = graph.finish().unwrap();
            stream.launch(&mut exec).unwrap();
            let mut replay = vec![half::f16::ZERO; t * width];
            stream
                .read_blocking(out.binding(), bytemuck::cast_slice_mut(&mut replay))
                .unwrap();
            assert_eq!(actual, replay);
            runtime.reset(&mut stream).unwrap();
            runtime
                .emit(
                    &mut Sink::Stream(&mut stream),
                    &mut profile,
                    0,
                    buffers[0].binding(),
                    buffers[1].binding(),
                    buffers[2].binding(),
                    out.binding(),
                    Some(gate_buffer.binding()),
                )
                .unwrap();
            stream
                .read_blocking(out.binding(), bytemuck::cast_slice_mut(&mut actual))
                .unwrap();
            let error = (actual
                .iter()
                .zip(&gated_expected)
                .map(|(a, e)| (a.to_f32() - e).powi(2))
                .sum::<f32>()
                / gated_expected.iter().map(|e| e * e).sum::<f32>())
            .sqrt();
            if let Some(dir) = &dump {
                std::fs::write(
                    dir.join(format!("gated-{bits}.f16")),
                    bytemuck::cast_slice(&actual),
                )
                .unwrap();
            }
            eprintln!("VSA gated {bits}: normalized RMSE={error}");
            assert!(error < if bits == 16 { 0.01 } else { 0.03 });
            // Every score ties at zero: the threshold comparison must keep every
            // tied block, including live rows at both ends of partial cubes.
            stream.fill(buffers[0].binding(), 0).unwrap();
            runtime
                .emit(
                    &mut Sink::Stream(&mut stream),
                    &mut profile,
                    0,
                    buffers[0].binding(),
                    buffers[1].binding(),
                    buffers[2].binding(),
                    out.binding(),
                    None,
                )
                .unwrap();
            let mut routes = vec![0i32; heads * nb * nb];
            stream
                .read_blocking(
                    runtime.routes.binding(),
                    bytemuck::cast_slice_mut(&mut routes),
                )
                .unwrap();
            assert!(routes.iter().all(|&r| r == 1));
            stream
                .upload(buffers[0].binding(), bytemuck::cast_slice(&data[0]))
                .unwrap();
        }
    }
}
