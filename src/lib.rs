#![doc = include_str!("../README.md")]

pub(crate) mod api;
pub(crate) mod macros;

pub mod environment;
pub mod error;
pub mod iobinding;
pub mod session;
pub mod value;

pub mod prelude {
    pub use super::environment::Environment;
    pub use super::error::{Error, Result};
    pub use super::iobinding::IoBinding;
    pub use super::session::Session;
    pub use super::value::{
        AllocatorDefault, AllocatorSession, AllocatorTrait, AsONNXTensorElementDataTypeTrait,
        ConstMemoryInfo, MemoryInfo, MemoryInfoTrait, ValueBorrowed, ValueTrait,
    };
    #[cfg(feature = "cuda")]
    pub use ortn_sys::cuda::{cudaError, cudaMemcpyKind};
    pub use ortn_sys::{
        GraphOptimizationLevel, OrtLoggingLevel, OrtMemType, OrtMemoryInfoDeviceType,
    };
}

#[cfg(test)]
#[test]
fn test_onnxruntime_run_ok() -> error::Result<()> {
    use ndarray::Array4;
    use ndarray_rand::{rand_distr::Uniform, RandomExt};
    use prelude::*;
    let _ = dotenv::dotenv();
    let _ = tracing_subscriber::fmt::try_init();

    // prepare input
    let inp = Array4::random([1, 1, 28, 28], Uniform::new(0., 1.));
    let inp = inp.view();
    let inp = ValueBorrowed::try_from(inp)?;

    #[allow(unused_mut)]
    let mut builder = Session::builder()
        .with_env(Environment::builder().with_name("minst").build()?)
        .with_graph_optimization_level(GraphOptimizationLevel::ORT_ENABLE_ALL)
        .with_intra_threads(4);

    #[cfg(feature = "cuda")]
    {
        builder = builder.with_use_tensor_rt(true).with_cuda_device(0);
    }

    let session = builder.build(include_bytes!("../models/mnist.onnx"))?;

    let out = session.run([inp])?.into_iter().next().expect("no output?");

    let out = match out.view::<f32>() {
        Ok(v) => v.to_owned(),
        Err(_) => out.clone_host()?.view::<f32>()?.to_owned(),
    };
    tracing::info!(?out);

    Ok(())
}

#[cfg(test)]
#[test]
fn test_onnxruntime_run_with_iobinding_ok() -> error::Result<()> {
    use ndarray::Array4;
    use ndarray_rand::{rand_distr::Uniform, RandomExt};
    use prelude::*;
    let _ = dotenv::dotenv();
    let _ = tracing_subscriber::fmt::try_init();

    // prepare input
    let inp = Array4::random([1, 1, 28, 28], Uniform::new(0., 1.));
    let inp = inp.view();
    let inp = ValueBorrowed::try_from(inp)?;

    #[allow(unused_mut)]
    let mut builder = Session::builder()
        .with_env(Environment::builder().with_name("minst").build()?)
        .with_graph_optimization_level(GraphOptimizationLevel::ORT_ENABLE_ALL)
        .with_intra_threads(4);

    #[cfg(feature = "cuda")]
    {
        builder = builder.with_use_cuda(true).with_cuda_device(0);
    }

    let session = builder.build(include_bytes!("../models/mnist.onnx"))?;

    let mut iobinding = IoBinding::new(&session)?;
    iobinding.bind_inputs([inp])?;

    #[allow(unused_mut)]
    let mut mem_info = MemoryInfo::new_cpu()?;

    #[cfg(feature = "cuda")]
    {
        mem_info = MemoryInfo::new_cuda(0)?;
    }

    iobinding.bind_outputs_to_device(&mem_info)?;

    session.run_with_iobinding(&mut iobinding)?;

    let outputs = iobinding.get_outputs()?;

    let out = outputs.values().get(0).unwrap();

    let out = match out.view::<f32>() {
        Ok(v) => v.to_owned(),
        Err(_) => out.clone_host()?.view::<f32>()?.to_owned(),
    };

    tracing::info!(?out);

    Ok(())
}
