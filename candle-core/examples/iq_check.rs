// Validation harness for the i-quant LOAD + CPU dequant path.
//
// For each of IQ2_XXS, IQ3_XXS, IQ4_XS, IQ1_S, IQ1_M it locates a real tensor of
// that dtype across the GLM GGUF shards, then checks:
//   (a) on-disk tensor byte length == n_blocks * type_size  (struct-size guard),
//   (b) dequantized values are all finite and not all-zero,
//   (c) the value range is sane for weights (within a few units of 0).
//
// A target dtype that does not appear in any shard is SKIPPED (not failed) — e.g.
// when a split GGUF is still downloading and the dtype lives in a later shard.
//
// Run with:
//   cargo run -p candle-core --example iq_check -- <dir-with-shards>

use candle_core::quantized::gguf_file::Content;
use candle_core::quantized::GgmlDType;
use candle_core::Device;
use std::fs::File;
use std::path::{Path, PathBuf};

fn shards(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .expect("read dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|e| e == "gguf").unwrap_or(false))
        .collect();
    v.sort();
    v
}

fn main() {
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/home/lukim/models/GLM-5.2-GGUF/UD-IQ1_M".to_string());
    let dir = Path::new(&dir);
    let device = Device::Cpu;

    let targets = [
        ("IQ2_XXS", GgmlDType::Iq2Xxs),
        ("IQ3_XXS", GgmlDType::Iq3Xxs),
        ("IQ4_XS", GgmlDType::Iq4Xs),
        ("IQ1_S", GgmlDType::Iq1S),
        ("IQ1_M", GgmlDType::Iq1M),
    ];

    // For each target dtype, find the first (shard, tensor) that uses it and
    // record per-dtype tensor counts across all shards.
    let mut found: std::collections::HashMap<GgmlDType, (PathBuf, String, candle_core::Shape)> =
        Default::default();
    let mut counts: std::collections::HashMap<GgmlDType, usize> = Default::default();

    let shard_paths = shards(dir);
    assert!(!shard_paths.is_empty(), "no .gguf shards found in {dir:?}");
    for shard in &shard_paths {
        let mut f = File::open(shard).expect("open shard");
        let content = Content::read(&mut f).expect("read gguf header");
        for (name, info) in content.tensor_infos.iter() {
            *counts.entry(info.ggml_dtype).or_default() += 1;
            for (_, dt) in targets.iter() {
                if info.ggml_dtype == *dt {
                    found
                        .entry(*dt)
                        .or_insert_with(|| (shard.clone(), name.clone(), info.shape.clone()));
                }
            }
        }
    }

    println!("== dtype tensor counts across {} shards ==", shard_paths.len());
    for (label, dt) in targets.iter() {
        println!("  {label:8} = {}", counts.get(dt).copied().unwrap_or(0));
    }
    println!();

    let mut all_ok = true;
    let mut n_checked = 0usize;
    for (label, dt) in targets.iter() {
        let (shard, name, shape) = match found.get(dt) {
            Some(t) => t,
            None => {
                // Not a failure: the dtype may live in a shard not yet downloaded.
                println!("[SKIP] {label}: no tensor of this dtype found in any shard\n");
                continue;
            }
        };
        n_checked += 1;

        let mut f = File::open(shard).expect("open shard");
        let content = Content::read(&mut f).expect("read gguf header");

        // (a) on-disk byte length == n_blocks * type_size.
        let elems = shape.elem_count();
        let block_size = dt.block_size();
        let type_size = dt.type_size();
        assert_eq!(elems % block_size, 0, "{label}: elems not divisible by block size");
        let n_blocks = elems / block_size;
        let expected_bytes = n_blocks * type_size;

        let qtensor = content.tensor(&mut f, name, &device).expect("read+load tensor");
        let on_disk_bytes = qtensor.storage_size_in_bytes();
        let byte_ok = on_disk_bytes == expected_bytes;

        // (b)+(c) dequantize and inspect.
        let t = qtensor.dequantize(&device).expect("dequantize");
        let v = t
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .expect("to_vec1 f32");
        let n = v.len();
        let n_finite = v.iter().filter(|x| x.is_finite()).count();
        let n_nonzero = v.iter().filter(|x| **x != 0.0).count();
        let (mut mn, mut mx) = (f32::INFINITY, f32::NEG_INFINITY);
        let mut sumsq = 0f64;
        for &x in &v {
            if x.is_finite() {
                mn = mn.min(x);
                mx = mx.max(x);
                sumsq += (x as f64) * (x as f64);
            }
        }
        let rms = (sumsq / n.max(1) as f64).sqrt();

        let all_finite = n_finite == n;
        let any_nonzero = n_nonzero > 0;
        // Weights after dequant are small; allow a generous bound.
        let sane_range = mn.is_finite() && mx.is_finite() && mn > -10.0 && mx < 10.0;

        let ok = byte_ok && all_finite && any_nonzero && sane_range;
        all_ok &= ok;

        println!("== {label} ==");
        println!("  shard   : {}", shard.file_name().unwrap().to_string_lossy());
        println!("  tensor  : {name}  shape={:?}", shape.dims());
        println!(
            "  bytes   : on_disk={on_disk_bytes} expected={expected_bytes} (n_blocks={n_blocks} type_size={type_size})  [{}]",
            if byte_ok { "OK" } else { "MISMATCH" }
        );
        println!(
            "  values  : n={n} finite={n_finite} nonzero={n_nonzero}  [{}]",
            if all_finite && any_nonzero { "OK" } else { "BAD" }
        );
        println!(
            "  range   : min={mn:.5} max={mx:.5} rms={rms:.5}  [{}]",
            if sane_range { "OK" } else { "OUT-OF-RANGE" }
        );
        println!("  RESULT  : {}\n", if ok { "PASS" } else { "FAIL" });
    }

    if all_ok {
        println!("ALL {n_checked} CHECKED I-QUANT TYPES: PASS");
    } else {
        println!("SOME I-QUANT TYPES FAILED");
        std::process::exit(1);
    }
}
