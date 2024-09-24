#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(clippy::all)]

#[cfg(all(target_os= "macos", target_arch = "aarch64"))]
include!("ffi/aarch64-macos-cpu.rs");

#[cfg(all(target_os= "linux", target_arch = "x86_64"))]
include!("ffi/x86_64-linux-cpu.rs");