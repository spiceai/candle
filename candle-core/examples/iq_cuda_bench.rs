// Throughput of IQ1_S dequantization, CPU reference vs the CUDA kernel.
//
// This is the dominant cost of serving an i-quant MoE: the dtype has no q8_1 mat-mul
// kernel, so every use of an expert weight dequantizes it first. Before the CUDA kernel
// existed that expansion ran on the host, which also meant copying the quantized bytes
// device->host and the f32 result back.
//
// Run with:
//   cargo run --release -p candle-core --features cuda --example iq_cuda_bench -- <dir-with-shards>

use candle_core::quantized::gguf_file::Content;
use candle_core::quantized::GgmlDType;
use candle_core::Device;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::Instant;

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
        .expect("usage: iq_cuda_bench <dir-with-gguf-shards>");
    // One expert's worth of weights: the unit the MoE forward actually touches.
    let experts: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let iters: usize = 5;
    let dtype = GgmlDType::Iq1S;
    let cuda = Device::new_cuda(0)?;

    for path in shards(Path::new(&dir)) {
        let mut file = File::open(&path).expect("open shard");
        let content = Content::read(&mut file).expect("read gguf");
        let mut names: Vec<_> = content
            .tensor_infos
            .iter()
            .filter(|(n, i)| i.ggml_dtype == dtype && n.contains("_exps."))
            .map(|(n, _)| n.clone())
            .collect();
        names.sort();
        let Some(name) = names.first() else { continue };

        let info = &content.tensor_infos[name];
        let dims = info.shape.dims();
        // [n_experts, out, in] -> take `experts` of them.
        let per_expert = info.shape.elem_count() / dims[0];
        let elems = per_expert * experts;
        let n_blocks = elems / dtype.block_size();

        let mut raw = vec![0u8; n_blocks * dtype.type_size()];
        std::io::Seek::seek(
            &mut file,
            std::io::SeekFrom::Start(content.tensor_data_offset + info.offset),
        )
        .expect("seek");
        std::io::Read::read_exact(&mut file, &mut raw).expect("read");

        println!(
            "{name}: {experts} expert(s), {} weights ({:.1} MB quantized -> {:.1} MB f32)",
            elems,
            raw.len() as f64 / 1e6,
            elems as f64 * 4.0 / 1e6
        );

        let mk = |dev: &Device| {
            candle_core::quantized::ggml_file::qtensor_from_ggml(dtype, &raw, vec![elems], dev)
        };

        // CPU reference: what the host path had to do for every expert use.
        let qt_cpu = mk(&Device::Cpu)?;
        let _ = qt_cpu.dequantize(&Device::Cpu)?; // warm
        let t = Instant::now();
        for _ in 0..iters {
            let _ = qt_cpu.dequantize(&Device::Cpu)?;
        }
        let cpu_s = t.elapsed().as_secs_f64() / iters as f64;

        // CUDA kernel.
        let qt_gpu = mk(&cuda)?;
        let _ = qt_gpu.dequantize(&cuda)?; // warm (also forces module load)
        cuda.synchronize()?;
        let t = Instant::now();
        for _ in 0..iters {
            let _ = qt_gpu.dequantize(&cuda)?;
        }
        cuda.synchronize()?;
        let gpu_s = t.elapsed().as_secs_f64() / iters as f64;

        println!(
            "  cpu  {:>9.2} ms   ({:.2} Gelem/s)",
            cpu_s * 1e3,
            elems as f64 / cpu_s / 1e9
        );
        println!(
            "  cuda {:>9.2} ms   ({:.2} Gelem/s)",
            gpu_s * 1e3,
            elems as f64 / gpu_s / 1e9
        );
        println!("  speedup {:.1}x", cpu_s / gpu_s);
        return Ok(());
    }
    println!("no IQ1_S expert tensor found in {dir}");
    Ok(())
}
