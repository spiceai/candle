#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::Result;
use candle::{test_utils, Device, Tensor};
use candle_nn::{LayerNorm, Module};

#[test]
fn layer_norm_initializes_from_an_empty_varmap() -> Result<()> {
    let device = &Device::Cpu;
    let varmap = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, candle::DType::F32, device);
    let ln = candle_nn::layer_norm(4, 1e-5, vb)?;
    assert_eq!(ln.weight().dims(), &[4]);
    assert_eq!(ln.bias().map(|bias| bias.dims().to_vec()), Some(vec![4]));
    Ok(())
}

#[test]
fn layer_norm_no_bias_loads_a_weight_only_checkpoint() -> Result<()> {
    let device = &Device::Cpu;
    let mut tensors = std::collections::HashMap::new();
    tensors.insert(
        "weight".to_string(),
        Tensor::ones((4,), candle::DType::F32, device)?,
    );
    let vb = candle_nn::VarBuilder::from_tensors(tensors, candle::DType::F32, device);
    let ln = candle_nn::layer_norm_no_bias(4, 1e-5, vb)?;
    assert!(ln.bias().is_none());
    assert_eq!(ln.weight().to_vec1::<f32>()?, [1.0, 1.0, 1.0, 1.0]);
    Ok(())
}

#[test]
fn layer_norm_loads_gamma_and_beta_names() -> Result<()> {
    let device = &Device::Cpu;
    let mut tensors = std::collections::HashMap::new();
    tensors.insert(
        "gamma".to_string(),
        Tensor::new(&[3f32, 3., 3., 3.], device)?,
    );
    tensors.insert(
        "beta".to_string(),
        Tensor::new(&[0.5f32, 0.5, 0.5, 0.5], device)?,
    );
    let vb = candle_nn::VarBuilder::from_tensors(tensors, candle::DType::F32, device);
    let ln = candle_nn::layer_norm(4, 1e-5, vb)?;
    assert_eq!(ln.weight().to_vec1::<f32>()?, [3.0, 3.0, 3.0, 3.0]);
    assert_eq!(
        ln.bias()
            .expect("affine layer norm has a bias")
            .to_vec1::<f32>()?,
        [0.5, 0.5, 0.5, 0.5]
    );
    Ok(())
}

#[test]
fn layer_norm() -> Result<()> {
    let device = &Device::Cpu;
    let w = Tensor::new(&[3f32], device)?;
    let b = Tensor::new(&[0.5f32], device)?;
    let ln2 = LayerNorm::new(Tensor::cat(&[&w, &w], 0)?, Tensor::cat(&[&b, &b], 0)?, 1e-8);
    let ln3 = LayerNorm::new(
        Tensor::cat(&[&w, &w, &w], 0)?,
        Tensor::cat(&[&b, &b, &b], 0)?,
        1e-8,
    );
    let ln = LayerNorm::new(w, b, 1e-8);
    assert_eq!(ln.eps(), 1e-8);
    assert!(ln.remove_mean());

    let two = Tensor::new(&[[[2f32]]], device)?;
    let res = ln.forward(&two)?.flatten_all()?;
    assert_eq!(res.to_vec1::<f32>()?, [0.5f32]);

    let inp = Tensor::new(&[[[4f32, 0f32]]], device)?;
    let res = ln2.forward(&inp)?;
    assert_eq!(res.to_vec3::<f32>()?, [[[3.5f32, -2.5]]]);

    let inp = Tensor::new(&[[[1f32, 2., 3.], [4., 5., 6.], [9., 8., 7.]]], device)?;
    let res = ln3.forward(&inp)?;
    assert_eq!(
        test_utils::to_vec3_round(&res, 4)?,
        [[
            [-3.1742, 0.5, 4.1742],
            [-3.1742, 0.5, 4.1742],
            [4.1742, 0.5, -3.1742]
        ]]
    );
    let mean = (res.sum_keepdim(2)? / 3.0)?;
    // The average value should be `b`.
    assert_eq!(
        test_utils::to_vec3_round(&mean, 4)?,
        [[[0.5], [0.5], [0.5]]]
    );
    let std = (res.broadcast_sub(&mean)?.sqr()?.sum_keepdim(2)?.sqrt()? / 3.0)?;
    // The standard deviation should be sqrt(`w`).
    assert_eq!(
        test_utils::to_vec3_round(&std, 4)?,
        [[[1.7321], [1.7321], [1.7321]]]
    );

    // Verify that rms_norm sets remove_mean to false.
    let rms = LayerNorm::rms_norm(Tensor::new(&[1f32], device)?, 1e-5);
    assert_eq!(rms.eps(), 1e-5);
    assert!(!rms.remove_mean());

    Ok(())
}
