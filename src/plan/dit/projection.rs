use super::*;
use crate::weights::{elem_bytes, pack_output_channels, packed_f32_rows, widen_one, Error};

/// The kernels use the regular, symmetric H256 rotation and one scale per row.
pub(super) fn validate_int8(ck: &Checkpoint, stem: &str, rows: usize, cols: usize) -> Result<()> {
    ck.at_checked(
        &format!("{stem}.weight"),
        Dtype::I8,
        &[rows as i64, cols as i64],
    )?;
    ck.at_checked(
        &format!("{stem}.weight_scale"),
        Dtype::F32,
        &[rows as i64, 1],
    )?;
    let name = format!("{stem}.comfy_quant");
    let entry = ck.at_checked(&name, Dtype::U8, &[-1])?;
    let metadata: serde_json::Value = serde_json::from_slice(ck.bytes(entry))
        .map_err(|e| Error::Layout(format!("{name}: {e}")))?;
    if metadata["format"] != "int8_tensorwise"
        || metadata["convrot"] != true
        || metadata["convrot_groupsize"] != 256
        || metadata.get("per_row").is_some_and(|v| v != true)
        || !cols.is_multiple_of(256)
    {
        return Err(Error::Layout(format!(
            "{name}: requires INT8 ConvRot with group size 256 and per-row scales"
        )));
    }
    Ok(())
}

/// Inverse ConvRot. The regular H4 is symmetric and its fourth Kronecker power
/// is H256; normalization is 1/16. This is not the Sylvester Hadamard convention.
fn unrotate(values: &mut [f32]) {
    for group in values.as_chunks_mut::<256>().0 {
        for stride in [1, 4, 16, 64] {
            for chunk in group.chunks_exact_mut(4 * stride) {
                for i in 0..stride {
                    let [a, b, c, d] = [
                        chunk[i],
                        chunk[i + stride],
                        chunk[i + 2 * stride],
                        chunk[i + 3 * stride],
                    ];
                    chunk[i] = a + b + c - d;
                    chunk[i + stride] = a + b - c + d;
                    chunk[i + 2 * stride] = a - b + c + d;
                    chunk[i + 3 * stride] = -a + b + c + d;
                }
            }
        }
        for value in group {
            *value *= 1.0 / 16.0;
        }
    }
}

/// BF16 row-major condition projection, or FP32 heads packed [N/32][K][32].
pub(super) fn rows(ck: &Checkpoint, parts: &[(&str, usize, usize)], bf16: bool) -> Result<Recipe> {
    let cols = parts[0].2;
    let outputs: usize = parts.iter().map(|p| p.1).sum();
    let target = if bf16 { Dtype::BF16 } else { Dtype::F32 };
    let mut names = Vec::new();
    let mut direct = true;
    for &(stem, rows, width) in parts {
        if width != cols || (!bf16 && !outputs.is_multiple_of(32)) {
            return Err(Error::Layout("incompatible projection shapes".into()));
        }
        let name = format!("{stem}.weight");
        let dtype = ck.at(&name)?.dtype;
        if dtype == Dtype::I8 {
            validate_int8(ck, stem, rows, cols)?;
        } else {
            float(ck, &name, &[rows as i64, cols as i64])?;
        }
        direct &= dtype == target;
        names.push(name);
    }
    if direct {
        let names: Vec<_> = names.iter().map(String::as_str).collect();
        return if bf16 {
            rows_of(ck, &names, 0)
        } else {
            packed_f32_rows(ck, &names)
        };
    }
    Ok(Recipe::Built {
        bytes: outputs * cols * if bf16 { 2 } else { 4 },
        build: Box::new(move |ck| {
            let mut bytes = Vec::with_capacity(outputs * cols * if bf16 { 2 } else { 4 });
            let mut row = vec![0.0f32; cols];
            for name in &names {
                let entry = ck.at(name)?;
                let source = ck.bytes(entry);
                let scales = if entry.dtype == Dtype::I8 {
                    Some(ck.bytes(ck.at(&format!("{}_scale", name))?))
                } else {
                    None
                };
                for r in 0..entry.rows() {
                    if let Some(scales) = scales {
                        let scale =
                            f32::from_le_bytes(scales[r * 4..r * 4 + 4].try_into().unwrap());
                        if !scale.is_finite() || scale < 0.0 {
                            return Err(Error::Layout(format!(
                                "{name}: invalid row scale {scale}"
                            )));
                        }
                        for (c, value) in row.iter_mut().enumerate() {
                            *value = (source[r * cols + c] as i8) as f32;
                        }
                        unrotate(&mut row);
                        for value in &mut row {
                            *value *= scale;
                        }
                    } else {
                        for (c, value) in row.iter_mut().enumerate() {
                            let offset = (r * cols + c) * elem_bytes(entry.dtype);
                            *value = widen_one(entry.dtype, &source[offset..])?;
                        }
                    }
                    for &value in &row {
                        if bf16 {
                            bytes.extend_from_slice(&half::bf16::from_f32(value).to_le_bytes());
                        } else {
                            bytes.extend_from_slice(&value.to_le_bytes());
                        }
                    }
                }
                ck.done_with(source);
            }
            Ok(if bf16 {
                bytes
            } else {
                pack_output_channels(&bytes, outputs, cols, 32)
            })
        }),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    // Dense Kronecker matrix oracle, independent of the radix-4 implementation.
    fn reference(codes: &[u8], col: usize) -> f32 {
        let h = [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]];
        let base = col / 256 * 256;
        (0..256)
            .map(|k| {
                let sign: i32 = (0..4)
                    .map(|digit| h[(k >> (digit * 2)) & 3][(col >> (digit * 2)) & 3])
                    .product();
                (codes[base + k] as i8) as f32 * sign as f32 / 16.0
            })
            .sum()
    }

    fn fixture(
        metadata: serde_json::Value,
        scale: f32,
    ) -> (tempfile::TempDir, Checkpoint, Vec<u8>) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("projection.safetensors");
        let codes: Vec<u8> = (0..32 * 512)
            .map(|i| ((i * 71 + i / 512 * 13) % 255) as u8)
            .collect();
        let scales: Vec<_> = (0..32)
            .flat_map(|r| (scale * (r + 1) as f32).to_le_bytes())
            .collect();
        let metadata = serde_json::to_vec(&metadata).unwrap();
        let floats: Vec<_> = (0..32 * 512)
            .flat_map(|i| half::bf16::from_f32((i % 13) as f32).to_le_bytes())
            .collect();
        let tensors = [
            ("q.weight", "I8", vec![32, 512], codes.clone()),
            ("q.weight_scale", "F32", vec![32, 1], scales),
            ("q.comfy_quant", "U8", vec![metadata.len()], metadata),
            ("f.weight", "BF16", vec![32, 512], floats),
        ];
        let mut header = serde_json::Map::new();
        let mut offset = 0;
        for (name, dtype, shape, data) in &tensors {
            header.insert(name.to_string(), serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + data.len()]}));
            offset += data.len();
        }
        let header = serde_json::to_vec(&header).unwrap();
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header).unwrap();
        for (_, _, _, data) in tensors {
            file.write_all(&data).unwrap();
        }
        drop(file);
        let ck = unsafe { Checkpoint::open(path) }.unwrap();
        (dir, ck, codes)
    }

    fn metadata() -> serde_json::Value {
        serde_json::json!({"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256, "per_row": true})
    }

    #[test]
    fn int8_projections_restore_rotation_scales_and_output_layout() {
        let (_dir, ck, codes) = fixture(metadata(), 0.0031);
        let bf = rows(&ck, &[("q", 32, 512)], true)
            .unwrap()
            .assemble(&ck)
            .unwrap();
        let packed = rows(&ck, &[("q", 32, 512), ("f", 32, 512)], false)
            .unwrap()
            .assemble(&ck)
            .unwrap();
        for row in 0..32 {
            for col in 0..512 {
                let want = reference(&codes[row * 512..(row + 1) * 512], col)
                    * (0.0031 * (row + 1) as f32);
                let offset = (col * 32 + row) * 4;
                let got = f32::from_le_bytes(packed[offset..offset + 4].try_into().unwrap());
                assert_eq!(got, want, "row {row}, col {col}");
                let offset = (row * 512 + col) * 2;
                assert_eq!(
                    &bf[offset..offset + 2],
                    &half::bf16::from_f32(want).to_le_bytes()
                );
                let offset = (32 * 512 + col * 32 + row) * 4;
                assert_eq!(
                    f32::from_le_bytes(packed[offset..offset + 4].try_into().unwrap()),
                    ((row * 512 + col) % 13) as f32
                );
            }
        }
    }

    #[test]
    fn unsupported_quantization_is_rejected_before_loading() {
        for (key, value) in [
            ("format", serde_json::json!("int8")),
            ("convrot", serde_json::json!(false)),
            ("convrot_groupsize", serde_json::json!(64)),
            ("per_row", serde_json::json!(false)),
        ] {
            let mut meta = metadata();
            meta[key] = value;
            let (_dir, ck, _) = fixture(meta, 1.0);
            let message = rows(&ck, &[("q", 32, 512)], true)
                .err()
                .unwrap()
                .to_string();
            assert!(message.contains("q.comfy_quant"), "{message}");
        }
        let mut legacy = metadata();
        legacy.as_object_mut().unwrap().remove("per_row");
        let (_dir, ck, _) = fixture(legacy, 1.0);
        assert!(rows(&ck, &[("q", 32, 512)], true).is_ok());
    }

    #[test]
    fn invalid_scales_fail_instead_of_poisoning_conditioning() {
        for scale in [-1.0, f32::NAN, f32::INFINITY] {
            let (_dir, ck, _) = fixture(metadata(), scale);
            assert!(rows(&ck, &[("q", 32, 512)], true)
                .unwrap()
                .assemble(&ck)
                .is_err());
        }
    }
}
