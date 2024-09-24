use std::ffi::CStr;
use std::ptr::null_mut;
use tracing::*;

use crate::api::API;
use crate::error::*;
use ortn_sys as ffi;
use smart_default::SmartDefault;

#[derive(Debug)]
pub struct Environment {
    inner: *mut ffi::OrtEnv,
}

unsafe impl Send for Environment {}
unsafe impl Sync for Environment {}


impl Environment {
    pub fn builder() -> EnvironmentBuilder {
        EnvironmentBuilder::default()
    }

    pub fn inner(&self) -> *mut ffi::OrtEnv {
        self.inner
    }

    pub fn into_arc(self) -> std::sync::Arc<Self> {
        std::sync::Arc::new(self)
    }
}

impl Drop for Environment
{
    fn drop(&mut self) {
        trace!("dropping {:?}", self);
        unsafe { API.ReleaseEnv.as_ref().expect("failed to get ReleaseSession")(self.inner) };
    }
}


#[derive(SmartDefault)]
pub struct EnvironmentBuilder {
    #[default(ffi::OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR)]
    logging_level: ffi::OrtLoggingLevel,
    #[default("default".to_string())]
    name: String,
}

impl EnvironmentBuilder {
    pub fn with_level(mut self, level: ffi::OrtLoggingLevel) -> Self {
        self.logging_level = level;
        self
    }

    pub fn with_name(mut self, name: impl ToString) -> Self {
        self.name = name.to_string();
        self
    }
}

#[allow(improper_ctypes_definitions)]
unsafe extern "C" fn logging_function(
    _param: *mut ::std::os::raw::c_void,
    severity: ffi::OrtLoggingLevel,
    category: *const ::std::os::raw::c_char,
    logid: *const ::std::os::raw::c_char,
    code_location: *const ::std::os::raw::c_char,
    message: *const ::std::os::raw::c_char,
) {
    let category = CStr::from_ptr(category);
    let logid = CStr::from_ptr(logid);
    let code_location = CStr::from_ptr(code_location);
    let message = CStr::from_ptr(message);
    match severity {
        ffi::OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE => trace!(?category, ?logid, ?code_location, ?message),
        ffi::OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO => trace!(?category, ?logid, ?code_location, ?message),
        ffi::OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING => warn!(?category, ?logid, ?code_location, ?message),
        ffi::OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR => error!(?category, ?logid, ?code_location, ?message),
        ffi::OrtLoggingLevel::ORT_LOGGING_LEVEL_FATAL => error!(?category, ?logid, ?code_location, ?message),
        _ => todo!(),
    }
}

impl EnvironmentBuilder {
    pub fn build(self) -> Result<Environment> {
        let mut inner = null_mut();
        let name = std::ffi::CString::new(self.name.as_bytes())?;
        rc(unsafe {
            API.CreateEnvWithCustomLogger
                .expect("failed to get CreateEnvWithCustomLogger")(
                Some(logging_function),
                null_mut(),
                self.logging_level,
                name.as_ptr(),
                &mut inner,
            )
        })?;
        Ok(Environment { inner })
    }
}
