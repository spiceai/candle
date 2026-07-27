// Equivalence check for the CUDA i-quant dequantization kernels.
//
// The CPU implementations in `quantized::k_quants` are the reference (they are direct ports
// of ggml's `dequantize_row_*`). For each supported dtype this loads a real tensor of that
// dtype out of a GGUF, dequantizes it on both devices, and compares elementwise — which is
// what catches a mistranscribed lattice table or an off-by-one in the bit unpacking, neither
// of which shows up as a compile error or an obviously wrong-looking tensor.
//
// Run with:
//   cargo run -p candle-core --features cuda --example iq_cuda_check -- <dir-with-shards>

use candle_core::quantized::gguf_file::Content;
use candle_core::quantized::GgmlDType;
use candle_core::Device;
use std::fs::File;
use std::path::{Path, PathBuf};

const TARGETS: &[GgmlDType] = &[
    GgmlDType::Iq1S,
    GgmlDType::Iq2Xxs,
    GgmlDType::Iq3Xxs,
    GgmlDType::Iq4Xs,
];

fn shards(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .expect("read dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|e| e == "gguf").unwrap_or(false))
        .collect();
    v.sort();
    v
}

fn main() -> candle_core::Result<()> {
    let dir = std::env::args()
        .nth(1)
        .expect("usage: iq_cuda_check <dir-with-gguf-shards>");
    let cuda = Device::new_cuda(0)?;

    let mut failures = 0usize;
    for &dtype in TARGETS {
        let mut checked = false;
        'shards: for path in shards(Path::new(&dir)) {
            let mut file = File::open(&path).expect("open shard");
            let content = Content::read(&mut file).expect("read gguf");
            // Sort for determinism: the same tensor should be picked run to run.
            let mut names: Vec<_> = content
                .tensor_infos
                .iter()
                .filter(|(_, i)| i.ggml_dtype == dtype)
                .map(|(n, _)| n.clone())
                .collect();
            names.sort();
            let Some(name) = names.first() else {
                continue;
            };

            // Read a bounded prefix rather than the whole tensor: a stacked expert weight is
            // billions of elements, and dequantizing all of it on both devices would need
            // tens of gigabytes for a check that a few thousand blocks already settles.
            let info = &content.tensor_infos[name];
            let block = dtype.block_size();
            let type_size = dtype.type_size();
            let rows = *info.shape.dims().last().expect("tensor has no dims");
            let blocks_per_row = rows / block;
            let want_blocks = (4096 / blocks_per_row.max(1)).max(1) * blocks_per_row;
            let n_blocks = want_blocks.min(info.shape.elem_count() / block);
            let elems = n_blocks * block;

            let mut raw = vec![0u8; n_blocks * type_size];
            std::io::Seek::seek(
                &mut file,
                std::io::SeekFrom::Start(content.tensor_data_offset + info.offset),
            )
            .expect("seek");
            std::io::Read::read_exact(&mut file, &mut raw).expect("read");

            let dq = |dev: &Device| -> candle_core::Result<Vec<f32>> {
                candle_core::quantized::ggml_file::qtensor_from_ggml(dtype, &raw, vec![elems], dev)?
                    .dequantize(dev)?
                    .flatten_all()?
                    .to_vec1::<f32>()
            };
            let cpu = dq(&Device::Cpu)?;
            let gpu = dq(&cuda)?;

            if cpu.len() != gpu.len() {
                println!(
                    "{dtype:?}: FAIL len {} (cpu) != {} (gpu)",
                    cpu.len(),
                    gpu.len()
                );
                failures += 1;
                checked = true;
                break 'shards;
            }

            let mut max_abs = 0f32;
            let mut first_bad = None;
            for (i, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
                let d = (c - g).abs();
                if d > max_abs {
                    max_abs = d;
                }
                // Both sides compute in f32 from identical integer inputs, so this should be
                // bit-exact; allow a hair of slack rather than asserting equality.
                if d > 1e-5 && first_bad.is_none() {
                    first_bad = Some((i, *c, *g));
                }
            }

            let nonzero = cpu.iter().filter(|v| **v != 0.0).count();
            match first_bad {
                None => println!(
                    "{dtype:?}: OK  {name} ({} values, {nonzero} nonzero, max|Δ| {max_abs:.3e})",
                    cpu.len()
                ),
                Some((i, c, g)) => {
                    println!(
                        "{dtype:?}: FAIL {name} at [{i}]: cpu {c} vs gpu {g} (max|Δ| {max_abs:.3e})"
                    );
                    failures += 1;
                }
            }
            checked = true;
            break 'shards;
        }
        if !checked {
            println!("{dtype:?}: SKIP (no tensor of this dtype found)");
        }
    }

    if failures > 0 {
        panic!("{failures} dtype(s) mismatched between CPU and CUDA");
    }
    println!("all checked dtypes match");
    Ok(())
}
