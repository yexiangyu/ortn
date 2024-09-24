#![doc = include_str!("../README.md")]

pub(crate) mod api;
pub mod environment;
pub mod error;
pub mod session;
pub mod value;

pub mod prelude {
    pub use super::environment::Environment;
    pub use super::error::{Error, Result};
    pub use super::session::Session;
    pub use super::value::{AsONNXTensorElementDataTypeTrait, ValueTrait, ValueView};
    pub use ortn_sys::{GraphOptimizationLevel, OrtLoggingLevel};
}

#[allow(unused)]
#[cfg(test)]
fn test_demo_code() -> error::Result<()> {
    use ndarray::Array4;
    use ndarray_rand::{rand_distr::Uniform, RandomExt};
    use prelude::*;
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
