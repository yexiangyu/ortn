use lazy_static::lazy_static;
use ortn_sys as ffi;

#[derive(Debug, Clone, Copy)]
pub struct Api {
    pub inner: *const ffi::OrtApi,
}

impl Default for Api
{
    fn default() -> Self {
        let base = unsafe { ffi::OrtGetApiBase() };
        let inner = unsafe {
            base.as_ref()
                .expect("failed to get api base")
                .GetApi
                .as_ref()
                .expect("failed to get api")(ffi::ORT_API_VERSION)
        };
        Self { inner }
    }
}

impl std::ops::Deref for Api {
    type Target = ffi::OrtApi;

    fn deref(&self) -> &Self::Target {
        unsafe { self.inner.as_ref().expect("failed to get api") }
    }
}

unsafe impl Send for Api {}
unsafe impl Sync for Api {}

lazy_static! {
    pub static ref API: Api = Api::default();
}
