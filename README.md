# `ortn`

Yet another ***minimum*** `rust` binding for `onnxruntime` `c_api`, inspired by [onnxruntime-rs](https://github.com/nbigaouette/onnxruntime-rs).

## What's ***"minimum"*** means? 

- Only subset of `c_api` is wrapped, enough to run a onnx model.
- Less `'a` lifetime generic...
- Less concept overhead when use `rust` compare to use `onnxruntime` `c_api`.
- ***Best effort*** to work with `latest` onnxruntime version on different platform, less `feature` flag introduced by multi-version of onnxruntime.
- Only shared library (`onnxruntime.dll, libonnxruntime.[so|dyn])` supported.

## Test Matrix

|   OS   | onnxuntime<br />version |  Arch  | CPU | CUDA | TensorRT | CANN |
| :-----: | :---------------------: | :-----: | :--: | :--: | :------: | :--: |
|   mac   |         1.19.2         | aarch64 |  ✅  | n/a |   n/a   | n/a |
|   mac   |         1.19.2         | intel64 |  ✅  | n/a |   n/a   | n/a |
|  linux  |         1.19.2         | intel64 |  ✅  |  ✅  |    ✅    | n/a |
| windows |          TODO          | intel64 | TODO | TODO |   TODO   | TODO |

## Getting Started

1. please download [`onnxruntime`](https://github.com/microsoft/onnxruntime) first, unzip, or build it from source. Binary downloaded from [release page](https://github.com/microsoft/onnxruntime/releases) is not signed, take care of it.

2. before start everything, setup environment variable to help `ortn` to find `header` or `libraries` needed.

- `ORT_LIB_DIR`: folder where `libonnxruntime.[so|dylib]` or `onnxruntime.dll` located
- `ORT_INC_DIR`: folder where header file: `onnxruntime/onnxruntime_c_api.h` located
- `DYLD_LIBRARY_PATH`: (mac only) folder where `libonnxruntime.dylib` located
- `LD_LIBRARY_PATH`: (linux only) folder where `libonnxruntime.so` located
- `PATH`: (windows only) folder where `onnxruntime.dll` located

3. build environment, build session, run session

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
Result::Ok(())
```

## Update bindings
In case bindings need to be update, just:

1. git clone this repo
```bash
git clone https://github.com/yexiangyu/ortn
```
2. export environment variable
```bash
export ORT_LIB_DIR=/path/to/onnxruntime/lib
export ORT_INC_DIR=/path/to/onnxruntime/include
```
3. build bindings with feature `bindgen` enabled
```bash
cargo build --features bindgen
```

## TODO

- More input data type like `f16`, `i64` ...
- More runtime provider like `rocm` and `cann`
- `onnxruntime-agi`
- `training api`