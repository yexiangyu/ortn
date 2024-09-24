# `ortn`

Yet another minimum `rust` binding for `onnxruntime` `c_api`.

## Limitations

- only shared library (`onnxruntime.dll, libonnxruntime.[so|dyn])` supported
- suppose to work with specified onnxruntime version on different platform only.

## TODO

- `IOBinding`
- More input data type like `f16`, `i64` ...
- More wrapped value type like `opencv::Mat`
- More runtime provider

## Supported Matrix

| OS      | version | Arch    | CPU  | CUDA | TensorRT | CANN |
| ------- | ------- | ------- | ---- | ---- | -------- | ---- |
| mac     | 1.17.1  | aarch64 | ✅   | n/a  | n/a      | n/a  |
| mac     | 1.17.1  | intel64 | TODO | n/a  | n/a      | n/a  |
| linux   | TODO    | intel64 | TODO | TODO | TODO     | TODO |
| windows | TODO    | intel64 | TODO | TODO | TODO     | TODO |

## Getting Started

1. before start everything, setup environment variable to help `ortn` to find `header` or `libraries` needed.

- `ORT_LIB_DIR`: folder where `libonnxruntime.[so|dylib]` located
- `ORT_INC_DIR`: folder where header file: `onnxruntime/onnxruntime_c_api.h` located

2. build environment, build session, run session

```rust
use ndarray::Array4;
use ndarray_rand::{rand_distr::Uniform, RandomExt};
use ortn::prelude::*;
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

    // output is a vector, we need to get first result
    .into_iter()
    .next()
    .unwrap()

    // view output as a f32 array
    .view::<f32>()?

    // the output is owned by session, copy it out as a owned tensor/ndarray
    .to_owned();

tracing::info!(?output);
```
