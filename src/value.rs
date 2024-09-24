use std::ffi::{c_void, CString};
use std::marker::PhantomData;
use std::ptr::null_mut;


use itertools::Itertools;
use ortn_sys::{self as ffi, ONNXTensorElementDataType};
use tracing::*;

use crate::api::API;
use crate::error::*;
use crate::session::{Session, TensorShapeInfo};
use ndarray::{ArrayView, ArrayViewD, Dimension};

/// all OrtValue type should implement this trait
pub trait ValueTrait {
    fn inner(&self) -> *mut ffi::OrtValue;

    fn as_ptr(&self) -> Result<*const c_void> {
        let mut ptr = null_mut();
        rc(unsafe {
            API.GetTensorMutableData
                .as_ref()
                .expect("failed to get GetTensorMutableData")(self.inner(), &mut ptr)
        })?;
        Ok(ptr)
    }

    fn as_mut_ptr(&mut self) -> Result<*mut c_void> {
        let ptr = self.as_ptr()? as *mut _;
        Ok(ptr)
    }

    fn shape_info(&self) -> Result<TensorShapeInfo> {
        let inner = self.inner();
        let mut tensor_type_ = null_mut();
        rc(unsafe {
            API.GetTensorTypeAndShape
                .as_ref()
                .expect("failed to GetTensorTypeAndShape")(
                inner as *const _, &mut tensor_type_
            )
        })?;

        let name = CString::new("")?;

        let mut dim_count = 0;

        rc(unsafe {
            API.GetDimensionsCount
                .as_ref()
                .expect("failed to get GetTensorShapeElementCount")(
                tensor_type_, &mut dim_count
            )
        })?;

        let mut dims = vec![0; dim_count];

        rc(unsafe {
            API.GetDimensions
                .as_ref()
                .expect("failed to get GetDimensions")(
                tensor_type_, dims.as_mut_ptr(), dim_count
            )
        })?;

        let mut data_type = ffi::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;

        rc(unsafe {
            API.GetTensorElementType
                .as_ref()
                .expect("failed to get GetTensorElementType")(
                tensor_type_, &mut data_type
            )
        })?;

        Ok(TensorShapeInfo {
            name,
            dims,
            data_type,
        })
    }

    fn view<T>(&self) -> Result<ArrayViewD<T>> {
        let ptr = self.as_ptr()? as *const T;
        let shape_info = self.shape_info()?;
        let dims = shape_info.dims.iter().map(|d| *d as usize).collect_vec();
        let dims = ndarray::IxDyn(&dims);
        Ok(unsafe { ArrayViewD::from_shape_ptr(dims, ptr) })
    }

}

/// value that created from extern data as reference
pub struct ValueView<'a, T> {
    inner: *mut ffi::OrtValue,
    _marker: PhantomData<&'a T>,
}

impl<'a, T> ValueTrait for ValueView<'a, T> {
    fn inner(&self) -> *mut ortn_sys::OrtValue {
        self.inner
    }
}

pub trait AsONNXTensorElementDataTypeTrait {
    fn typ() -> ONNXTensorElementDataType;
}

macro_rules! impl_as_onnx_tensor_element_data_type {
    ($t: ty, $ot: ident) => {
        impl AsONNXTensorElementDataTypeTrait for $t {
            fn typ() -> ONNXTensorElementDataType {
                ONNXTensorElementDataType::$ot
            }
        }
    };
}

impl_as_onnx_tensor_element_data_type!(f32, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

impl<'a, A, D> TryFrom<ArrayView<'a, A, D>> for ValueView<'a, ArrayView<'a, A, D>>
where
    A: AsONNXTensorElementDataTypeTrait,
    D: Dimension,
{
    type Error = crate::error::Error;

    fn try_from(value: ArrayView<'a, A, D>) -> std::result::Result<Self, Self::Error> {
        let typ = A::typ();
        let mut mem_info = null_mut();
        rc(unsafe {
            API.CreateCpuMemoryInfo
                .as_ref()
                .expect("failed to get CreateCpuMemoryInfo")(
                ffi::OrtAllocatorType::OrtArenaAllocator,
                ffi::OrtMemType::OrtMemTypeDefault,
                &mut mem_info,
            )
        })?;
        let dims = value.shape().iter().map(|d| *d as i64).collect_vec();
        let mut inner = null_mut();
        rc(unsafe {
            API.CreateTensorWithDataAsOrtValue
                .as_ref()
                .expect("failed to get CreateTensorWithDataAsOrtValue")(
                mem_info as *const _,
                value.as_ptr() as *mut _,
                value.len() * size_of::<A>(),
                dims.as_ptr(),
                dims.len(),
                typ,
                &mut inner,
            )
        })?;
        Ok(ValueView {
            inner,
            _marker: PhantomData,
        })
    }
}

/// value that created by and managed by session, like output
pub struct ValueOutput<'a> {
    inner: *mut ffi::OrtValue,
    _session: PhantomData<&'a Session>,
}

impl<'a> ValueTrait for ValueOutput<'a> {
    fn inner(&self) -> *mut ortn_sys::OrtValue {
        self.inner
    }
}

impl<'a> ValueOutput<'a> {
    pub fn new(inner: *mut ffi::OrtValue, _session: &'a Session) -> Self {
        Self {
            inner,
            _session: PhantomData,
        }
    }
}

#[derive(Debug)]
pub struct MemoryInfo {
    pub inner: *mut ffi::OrtMemoryInfo,
}

impl MemoryInfo {
    pub fn new_cpu() -> Result<Self> {
        let mut inner = null_mut();
        rc(unsafe {
            API.CreateCpuMemoryInfo
                .as_ref()
                .expect("failed to get CreateCpuMemoryInfo")(
                ffi::OrtAllocatorType::OrtArenaAllocator,
                ffi::OrtMemType::OrtMemTypeDefault,
                &mut inner,
            )
        })?;
        Ok(MemoryInfo { inner })
    }

    pub fn new_cuda(device_id: u32) -> Result<Self> {
        let name = CString::new("CUDA")?;
        let mut inner = null_mut();
        rc(unsafe {
            API.CreateMemoryInfo
                .as_ref()
                .expect("failed to get CreateMemoryInfo")(
                name.as_ptr(),
                ffi::OrtAllocatorType::OrtDeviceAllocator,
                device_id as i32,
                ffi::OrtMemType::OrtMemTypeDefault,
                &mut inner,
            )
        })?;
        Ok(MemoryInfo { inner })
    }
}

impl Drop for MemoryInfo {
    fn drop(&mut self) {
        trace!("dropping {:?}", self);
        unsafe {
            API.ReleaseMemoryInfo
                .as_ref()
                .expect("failed to get ReleaseMemoryInfo")(self.inner);
        }
    }
}

pub trait AllocatorTrait: std::fmt::Debug {
    fn inner(&self) -> *mut ffi::OrtAllocator;
}

#[derive(Debug)]
pub struct AllocatorDefault {
    pub inner: *mut ffi::OrtAllocator,
}

impl AllocatorTrait for AllocatorDefault {
    fn inner(&self) -> *mut ffi::OrtAllocator {
        self.inner
    }
}

impl Drop for AllocatorDefault {
    fn drop(&mut self) {
        trace!("dropping {:?}", self);
    }
}

impl AllocatorDefault {
    pub fn new() -> Result<Self> {
        let mut inner = null_mut();
        rc(unsafe {
            API.GetAllocatorWithDefaultOptions
                .as_ref()
                .expect("failed to get GetAllocatorWithDefaultOptions")(&mut inner)
        })?;
        Ok(AllocatorDefault { inner })
    }
}


#[derive(Debug)]
pub struct AllocatorSession<'a> {
    pub inner: *mut ffi::OrtAllocator,
    pub session: &'a Session,
}

impl <'a> AllocatorSession<'a> {

    pub fn new(memory_info: &MemoryInfo, session: &'a Session) -> Result<Self> {
        let mut inner = null_mut();
        rc(unsafe {
            API.CreateAllocator
                .as_ref()
                .expect("failed to get CreateAllocator")(session.inner, memory_info.inner, &mut inner)
        })?;
        Ok(AllocatorSession { inner, session })
    }
}

impl Drop for AllocatorSession<'_> {
    fn drop(&mut self) {
        trace!("dropping {:?}", self);
        unsafe {
            API.ReleaseAllocator
                .as_ref()
                .expect("failed to get ReleaseAllocator")(self.inner);
        }
    }
}

#[derive(Debug)]
pub struct ValueAllocated<'a,  A> where A: AllocatorTrait {
    pub inner: *mut ffi::OrtValue,
    pub allocator: &'a A,
}

impl <'a, A> ValueTrait for ValueAllocated<'a, A> where A: AllocatorTrait {
    fn inner(&self) -> *mut ffi::OrtValue {
        self.inner
    }
}

impl <'a, A> Drop for ValueAllocated<'a, A> where A: AllocatorTrait {
    fn drop(&mut self) {
        trace!("dropping {:?}", self);
        unsafe {
            API.AllocatorFree.as_ref().expect("failed to get AllocatorFree")(self.allocator.inner(), self.inner as *mut _);
        }
    }
}
