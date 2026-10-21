//! H3 checkpoints, including merges with BF16 DiT blocks and INT8 refiners/heads.
//! Transformer matrices retain their stored precision. Small projections are restored
//! to the unrotated floating-point layout consumed by the embedding/head kernels.
mod projection;

use crate::checkpoint::Checkpoint;
use crate::model::*;
use crate::weights::{
    interleave16, rows_of, scales_interleave16, scales_rows, widen_f32, Recipe, Result,
};
use hrx::artifacts::safetensors::DType as Dtype;
use std::collections::BTreeMap;

type Table = BTreeMap<String, Recipe>;

fn float(ck: &Checkpoint, name: &str, shape: &[i64]) -> Result<String> {
    let dtype = ck.at(name)?.dtype;
    if !matches!(dtype, Dtype::F32 | Dtype::F16 | Dtype::BF16) {
        return Err(crate::weights::Error::Layout(format!(
            "{name}: expected floating-point weights, found {dtype:?}"
        )));
    }
    ck.at_checked(name, dtype, shape)?;
    Ok(name.into())
}

fn vector(ck: &Checkpoint, out: &mut Table, dest: &str, src: &str, len: usize) -> Result<()> {
    out.insert(
        dest.into(),
        widen_f32(ck, &[&float(ck, src, &[len as i64])?])?,
    );
    Ok(())
}

fn block(ck: &Checkpoint, out: &mut Table, src: &str, dest: &str) -> Result<()> {
    let dtype = ck.at(&format!("{src}attn.qkv_proj.weight"))?.dtype;
    if !matches!(dtype, Dtype::I8 | Dtype::BF16) {
        return Err(crate::weights::Error::Layout(format!(
            "{src}: expected INT8 ConvRot or BF16 blocks, found {dtype:?}"
        )));
    }
    let quant = dtype == Dtype::I8;
    for (op, target, rows, cols) in [
        ("attn.qkv_proj", "qkv", QKV, HID),
        ("attn.out_proj", "out", HID, INNER),
        ("mlp.fc1", "gu", 2 * FFN, HID),
        ("mlp.fc2", "down", HID, FFN),
    ] {
        let stem = format!("{src}{op}");
        let name = format!("{stem}.weight");
        ck.at_checked(&name, dtype, &[rows as i64, cols as i64])?;
        let pitch = gemm_pitch(cols, if quant { 8 } else { 16 }) * if quant { 1 } else { 2 };
        let recipe = if target == "gu" {
            interleave16(ck, (&name, 0), (&name, FFN), FFN, pitch)?
        } else {
            rows_of(ck, &[&name], pitch)?
        };
        out.insert(format!("{dest}{target}.q"), recipe);
        if quant {
            projection::validate_int8(ck, &stem, rows, cols)?;
            let scale = format!("{stem}.weight_scale");
            out.insert(
                format!("{dest}{target}.s"),
                if target == "gu" {
                    scales_interleave16(ck, (&scale, 0), (&scale, FFN), FFN)?
                } else {
                    scales_rows(ck, &[&scale])?
                },
            );
        }
    }
    for (target, source, len) in [
        ("norm1", "norm1.weight", HID),
        ("norm2", "norm2.weight", HID),
        ("qnorm", "attn.q_norm.weight", HEAD_DIM),
        ("knorm", "attn.k_norm.weight", HEAD_DIM),
    ] {
        vector(
            ck,
            out,
            &format!("{dest}{target}"),
            &format!("{src}{source}"),
            len,
        )?;
    }
    Ok(())
}

pub fn plan(ck: &Checkpoint, out: &mut Table) -> Result<()> {
    for i in 0..BLOCKS {
        let p = format!("blocks.{i}.");
        block(ck, out, &p, &p)?;
        adaln(
            ck,
            out,
            &p,
            &format!("h3.blocks.{i}."),
            MODALITIES * 6 * HID,
        )?;
    }
    for i in 0..REFINER_BLOCKS {
        block(
            ck,
            out,
            &format!("token_refiner.blocks.{i}."),
            &format!("h3.refiner.{i}."),
        )?;
    }
    out.insert(
        "h3.adaln_t_table".into(),
        widen_f32(ck, &[&float(ck, "adaln_t_table", &[1025, 8])?])?,
    );
    vector(ck, out, "h3.rope_inv_freq", "rope.inv_freq", 16)?;
    adaln(ck, out, "final_layer.", "h3.final.", 2 * HID)?;
    vector(ck, out, "h3.final.norm", "final_layer.norm.weight", HID)?;
    vector(
        ck,
        out,
        "h3.refiner.final_norm",
        "token_refiner.final_norm.weight",
        HID,
    )?;

    for (dest, src, rows, cols, bf16) in [
        ("h3.cond", "condition_proj", HID, TEXT_DIM, true),
        ("h3.video_in", "video_patch_proj", HID, VIDEO_PATCH, false),
        ("h3.audio_in", "audio_patch_proj", HID, AUDIO_CH, false),
    ] {
        out.insert(
            format!("{dest}.w"),
            projection::rows(ck, &[(src, rows, cols)], bf16)?,
        );
        vector(ck, out, &format!("{dest}.b"), &format!("{src}.bias"), rows)?;
    }
    out.insert(
        "h3.final.out.w".into(),
        projection::rows(
            ck,
            &[
                ("final_layer.video_out", VIDEO_PATCH, HID),
                ("final_layer.audio_out", AUDIO_CH, HID),
            ],
            false,
        )?,
    );
    out.insert(
        "h3.final.out.b".into(),
        widen_f32(
            ck,
            &[
                &float(ck, "final_layer.video_out.bias", &[VIDEO_PATCH as i64])?,
                &float(ck, "final_layer.audio_out.bias", &[AUDIO_CH as i64])?,
            ],
        )?,
    );
    Ok(())
}

fn adaln(ck: &Checkpoint, out: &mut Table, src: &str, dest: &str, rows: usize) -> Result<()> {
    out.insert(
        format!("{dest}adaln.w"),
        widen_f32(
            ck,
            &[&float(
                ck,
                &format!("{src}adaln_proj.linear.weight"),
                &[rows as i64, 8],
            )?],
        )?,
    );
    vector(
        ck,
        out,
        &format!("{dest}adaln.b"),
        &format!("{src}adaln_proj.linear.bias"),
        rows,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Seek, SeekFrom, Write};

    // Sparse payloads exercise the real H3 dimensions without allocating model weights.
    fn checkpoint(dtype: Dtype, mixed: bool) -> (tempfile::TempDir, Checkpoint) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("block.safetensors");
        let mut header = serde_json::Map::new();
        let mut offset = 0usize;
        let mut descriptors = Vec::new();
        let mut add = |name: String, dtype: Dtype, shape: Vec<usize>, data: Option<Vec<u8>>| {
            let bytes = shape.iter().product::<usize>() * crate::weights::elem_bytes(dtype);
            header.insert(name, serde_json::json!({"dtype": format!("{dtype:?}"), "shape": shape, "data_offsets": [offset, offset + bytes]}));
            if let Some(data) = data {
                descriptors.push((offset, data));
            }
            offset += bytes;
        };
        for (op, rows, cols) in [
            ("attn.qkv_proj", QKV, HID),
            ("attn.out_proj", HID, INNER),
            ("mlp.fc1", 2 * FFN, HID),
            ("mlp.fc2", HID, FFN),
        ] {
            let actual = if mixed && op == "mlp.fc2" {
                Dtype::BF16
            } else {
                dtype
            };
            add(
                format!("blocks.0.{op}.weight"),
                actual,
                vec![rows, cols],
                None,
            );
            if dtype == Dtype::I8 {
                add(
                    format!("blocks.0.{op}.weight_scale"),
                    Dtype::F32,
                    vec![rows, 1],
                    None,
                );
                let data =
                    br#"{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}"#
                        .to_vec();
                add(
                    format!("blocks.0.{op}.comfy_quant"),
                    Dtype::U8,
                    vec![data.len()],
                    Some(data),
                );
            }
        }
        for (name, len) in [
            ("norm1", HID),
            ("norm2", HID),
            ("attn.q_norm", HEAD_DIM),
            ("attn.k_norm", HEAD_DIM),
        ] {
            add(
                format!("blocks.0.{name}.weight"),
                Dtype::BF16,
                vec![len],
                None,
            );
        }
        let header = serde_json::to_vec(&header).unwrap();
        let base = header.len() as u64 + 8;
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header).unwrap();
        file.set_len(base + offset as u64).unwrap();
        for (offset, data) in descriptors {
            file.seek(SeekFrom::Start(base + offset as u64)).unwrap();
            file.write_all(&data).unwrap();
        }
        drop(file);
        let ck = unsafe { Checkpoint::open(path) }.unwrap();
        (dir, ck)
    }

    #[test]
    fn blocks_keep_their_stored_precision_and_gate_up_order() {
        for dtype in [Dtype::BF16, Dtype::I8] {
            let (_dir, ck) = checkpoint(dtype, false);
            let mut out = Table::new();
            block(&ck, &mut out, "blocks.0.", "h3.refiner.0.").unwrap();
            assert_eq!(out.contains_key("h3.refiner.0.qkv.s"), dtype == Dtype::I8);
            let Recipe::Rows {
                row_bytes,
                pitch_bytes,
                segments,
                ..
            } = &out["h3.refiner.0.gu.q"]
            else {
                panic!("matrix must remain a row recipe")
            };
            let bytes = crate::weights::elem_bytes(dtype);
            assert_eq!(*row_bytes, HID * bytes);
            assert_eq!(*pitch_bytes, gemm_pitch(HID, bytes * 8) * bytes);
            assert_eq!((segments[0].row0, segments[0].rows), (0, 16));
            assert_eq!((segments[1].row0, segments[1].rows), (FFN, 16));
        }
    }

    #[test]
    fn a_block_with_inconsistent_matrix_precision_fails_at_open() {
        let (_dir, ck) = checkpoint(Dtype::I8, true);
        let message = block(&ck, &mut Table::new(), "blocks.0.", "blocks.0.")
            .unwrap_err()
            .to_string();
        assert!(message.contains("blocks.0.mlp.fc2.weight"), "{message}");
        assert!(
            message.contains("BF16") && message.contains("I8"),
            "{message}"
        );
    }
}
