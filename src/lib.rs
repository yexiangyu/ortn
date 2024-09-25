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
    pub use super::session::Session;
    pub use super::iobinding::IoBinding;
    pub use super::value::{
        AllocatorDefault, AllocatorSession, AllocatorTrait, AsONNXTensorElementDataTypeTrait,
        MemoryInfo, ValueTrait, ValueView,
    };
    pub use ortn_sys::{GraphOptimizationLevel, OrtLoggingLevel};
}

#[allow(unused)]
#[cfg(test)]
// #[test]
fn test_demo_code() -> error::Result<()> {
    use ndarray::Array4;
    use ndarray_rand::{rand_distr::Uniform, RandomExt};
    use prelude::*;
    let _ = dotenv::dotenv();
    std::env::set_var("RUST_LOG", "trace");

    let _ = tracing_subscriber::fmt::try_init();

    let output = Session::builder()
        // create env and use it as session's env
        .with_env(
            Environment::builder()
                .with_name("minst")
                .with_level(OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE)
                .build()?,
        )
        // use cuda
        // .with_use_tensor_rt(true)
        // disable all optimization
        .with_graph_optimization_level(GraphOptimizationLevel::ORT_DISABLE_ALL)
        // set session intra threads to 4
        .with_intra_threads(4)
        // build model
        .build(include_bytes!("../models/mnist.onnx"))?
        // run model
        .run([
            // convert input tensor to ValueView
            ValueView::try_from(
                // create random input tensor
                Array4::random([1, 1, 28, 28], Uniform::new(0., 1.)).view(),
            )?,
        ])?
        // output is a vector
        .into_iter()
        // get first output
        .next()
        .unwrap()
        // view output as a f32 array
        .view::<f32>()?
        // the output is owned by session, copy it out as a owned tensor/ndarray
        .to_owned();

    tracing::info!(?output);

    Ok(())
}

#[allow(unused)]
#[cfg(test)]
#[test]
fn test_demo_code_iobinding() -> error::Result<()> {
    use ndarray::{Array2, Array4};
    use ndarray_rand::{rand_distr::Uniform, RandomExt};
    use prelude::*;

    let _ = dotenv::dotenv();

    std::env::set_var("RUST_LOG", "trace");

    let _ = tracing_subscriber::fmt::try_init();

    let session = Session::builder()
        // create env and use it as session's env
        .with_env(
            Environment::builder()
                .with_name("minst")
                .with_level(OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE)
                .build()?,
        )
        // use cuda
        // .with_use_tensor_rt(true)
        // disable all optimization
        .with_graph_optimization_level(GraphOptimizationLevel::ORT_DISABLE_ALL)
        // set session intra threads to 4
        .with_intra_threads(4)
        // build model
        .build(include_bytes!("../models/mnist.onnx"))?;

    let input = Array4::random([1, 1, 28, 28], Uniform::new(0., 1.));
    let mut input = ValueView::try_from(input.view())?;

    let output_ =  Array2::<f32>::zeros([1, 10]);
    let mut output = ValueView::try_from(output_.view())?;

    let allocator = AllocatorDefault::new()?;
    let memory_info = MemoryInfo::new_cpu()?;

    let mut io_binding = IoBinding::new(&session)?;

    io_binding.bind_input(0, &mut input)?;
    io_binding.bind_output_to_device(0, &memory_info)?;
    // run model
    session.run_with_iobinding(&mut io_binding)?;

    let output = io_binding.outputs(&allocator)?.into_iter().next().unwrap();
    let output = output.view::<f32>()?;

    tracing::info!(?output);

    Ok(())
}
