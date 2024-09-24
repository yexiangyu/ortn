#![allow(unreachable_code)]

static ORT_LIB_DIR: &str = "ORT_LIB_DIR";
static ORT_INC_DIR: &str = "ORT_INC_DIR";

#[cfg(target_arch = "aarch64")]
pub static TARGET_ARCH: &str = "aarch64";

#[cfg(target_arch = "x86_64")]
pub static TARGET_ARCH: &str = "x86_64";

#[cfg(target_os = "macos")]
pub static TARGET_OS: &str = "macos";

#[cfg(target_os = "linux")]
pub static TARGET_OS: &'static str = "linux";

#[cfg(target_os = "windows")]
pub static TARGET_OS: &'static str = "windows";

#[cfg(feature = "cuda")]
pub static ACCL: &'static str = "cuda";

#[cfg(not(feature = "cuda"))]
pub static ACCL: &str = "cpu";

fn main() {
    let _ = dotenv::dotenv();

    let ort_lib_dir = std::env::var(ORT_LIB_DIR).expect("could not find ORT_LIB_DIR env var");
    println!("cargo::rerun-if-env-changed={}", ORT_LIB_DIR);
    println!("cargo::rustc-link-search={}", ort_lib_dir);
    println!("cargo::rustc-link-lib=onnxruntime");

    #[cfg(not(feature = "bindgen"))]
    {
        return;
    }

    println!("cargo:rerun-if-env-changed={}", ORT_INC_DIR);
    let ort_inc_dir = std::env::var(ORT_INC_DIR).expect("could not find ORT_INC_DIR env var");

    let triplet = format!("{}-{}-{}", TARGET_ARCH, TARGET_OS, ACCL);

    let bind_output = triplet + ".rs";

    let mut header = format!("{}/onnxruntime/onnxruntime_c_api.h", ort_inc_dir);
    if !std::path::PathBuf::from(&header).exists() {
        header = format!("{}/onnxruntime_c_api.h", ort_inc_dir);
    }

    #[cfg(all(target_os = "macos", feature = "cuda"))]
    {
        panic!("CUDA is not supported on MacOS");
    }

    bindgen::builder()
        .header(header)
        .default_enum_style(bindgen::EnumVariation::Rust {
            non_exhaustive: true,
        })
        .clang_arg(format!("-I{}", ort_inc_dir))
        .generate()
        .expect("could not generate bindings")
        .write_to_file(format!("src/ffi/{}", bind_output))
        .expect("could not write bindings");
}
