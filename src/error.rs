use crate::api::API;
use ortn_sys as ffi;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    NullError(#[from] std::ffi::NulError),
    #[error("code={:?}, message={}", .code, .message)]
    ApiError {
        code: ffi::OrtErrorCode,
        message: String,
    }
}

pub type Result<T> = std::result::Result<T, Error>;

pub fn rc(status: *mut ffi::OrtStatus) -> Result<()> {
    if status.is_null() {
        return Ok(());
    }
    let code = unsafe { API.GetErrorCode.as_ref().unwrap()(status) };
    let message = unsafe { API.GetErrorMessage.as_ref().unwrap()(status) };
    let message = unsafe { std::ffi::CStr::from_ptr(message) };
    let message = message.to_str().unwrap_or("!!!NON-UTF8!!!").to_string();
    Err(Error::ApiError { code, message })
}
