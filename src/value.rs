use std::ffi::{c_void, CStr, CString};
use std::marker::PhantomData;
use std::ptr::{null, null_mut};

use itertools::Itertools;
use ortn_sys::{self as ffi, ONNXTensorElementDataType};
use tracing::*;

use crate::error::*;
use crate::macros::*;
use crate::session::{Session, TensorShapeInfo};
use lazy_static::lazy_static;
use ndarray::{ArrayView, ArrayViewD, Dimension};

lazy_static! {
    pub static ref ALLOCATOR_DEFAULT: AllocatorDefault =
        AllocatorDefault::new().expect("failed to create default allocator");
}

pub trait ONNXTensorElementDataTypeSizeTrait {
    fn size(&self) -> usize;
}

impl ONNXTensorElementDataTypeSizeTrait for ONNXTensorElementDataType {
    fn size(&self) -> usize {
        match self {
            ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT => 4,
            ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 => 4,
            ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 => 8,
            ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 => 1,
            ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 => 1,
            ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16 => 2,
            ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 => 2,
            ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING => 1,
            ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL => 1,
            ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 => 2,
            ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE => 8,
            ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32 => 4,
            ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64 => 8,
            ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64 => 8,
            ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128 => 16,
            ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 => 2,
            ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN => 1,
            ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ => 1,
            ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2 => 1,
            ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ => 1,
            ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4 => 0, //WTF?
            ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4 => 0,  //WTF?
            ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED => todo!(),
            _ => todo!(),
        }
    }
}

/// all OrtValue type should implement this trait
pub trait ValueTrait {
    fn inner(&self) -> *mut ffi::OrtValue;

    fn as_ptr(&self) -> Result<*const c_void> {
        let mut ptr = null_mut();
        rc(call_api!(GetTensorMutableData, self.inner(), &mut ptr))?;
        Ok(ptr)
    }

    fn as_mut_ptr(&mut self) -> Result<*mut c_void> {
        let ptr = self.as_ptr()? as *mut _;
        Ok(ptr)
    }

    fn shape_info(&self) -> Result<TensorShapeInfo> {
        let inner = self.inner();
        let mut tensor_type_ = null_mut();
        rc(call_api!(GetTensorTypeAndShape, inner, &mut tensor_type_))?;
        let name = CString::new("")?;

        let mut dim_count = 0;

        rc(call_api!(GetDimensionsCount, tensor_type_, &mut dim_count))?;

        let mut dims = vec![0; dim_count];

        rc(call_api!(
            GetDimensions,
            tensor_type_,
            dims.as_mut_ptr(),
            dim_count
        ))?;

        let mut data_type = ffi::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;

        rc(call_api!(
            GetTensorElementType,
            tensor_type_,
            &mut data_type
        ))?;

        Ok(TensorShapeInfo {
            name,
            dims,
            data_type,
        })
    }

    fn mem_info(&self) -> Result<ConstMemoryInfo> {
        let mut mem_info = null();
        rc(call_api!(GetTensorMemoryInfo, self.inner(), &mut mem_info))?;
        Ok(ConstMemoryInfo { inner: mem_info })
    }

    fn view<T>(&self) -> Result<ArrayViewD<T>> {
        match self.mem_info()?.device_type()? {
            ffi::OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_CPU => {}
            _ => return Err(Error::ValueViewOnDeviceSide),
        }
        let ptr = self.as_ptr()? as *const T;
        let shape_info = self.shape_info()?;
        let dims = shape_info.dims.iter().map(|d| *d as usize).collect_vec();
        let dims = ndarray::IxDyn(&dims);
        Ok(unsafe { ArrayViewD::from_shape_ptr(dims, ptr) })
    }

    fn assign(&self, rhs: &mut impl ValueTrait) -> Result<()> {
        let shape_info = self.shape_info()?;
        let typ = shape_info.data_type;
        let dims = shape_info.dims;
        let n_elements = dims.iter().fold(1, |acc, d| acc * d) as usize;
        let size = n_elements * typ.size();

        let mem_info_lhs = self.mem_info()?;
        let mem_info_rhs = rhs.mem_info()?;

        let mem_name_lhs = mem_info_lhs.name()?;
        let mem_name_rhs = mem_info_rhs.name()?;

        let dst = rhs.as_mut_ptr()? as *mut _;
        let src = self.as_ptr()? as *const _;

        trace!(
            "copy from {} => {} n_elements={}, typ={:?}",
            mem_name_lhs,
            mem_name_rhs,
            n_elements,
            typ
        );

        match (mem_info_lhs.device_type()?, mem_info_rhs.device_type()?) {
            (
                ortn_sys::OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_CPU,
                ortn_sys::OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_CPU,
            ) => {
                unsafe { std::ptr::copy_nonoverlapping(src, dst, size) };
            }
            (
                ortn_sys::OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_CPU,
                ortn_sys::OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_GPU,
            ) => match mem_name_rhs.as_str() {
                #[cfg(feature = "cuda")]
                "Cuda" => {
                    cuda::cuda_mem_copy(
                        dst,
                        src,
                        size,
                        ffi::cuda::cudaMemcpyKind::cudaMemcpyHostToDevice,
                    )?;
                }
                _ => {
                    error!(
                        "failed to copy data from {} => {}, not implemented",
                        mem_name_lhs, mem_name_rhs
                    );
                    todo!()
                }
            },
            (
                ortn_sys::OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_GPU,
                ortn_sys::OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_CPU,
            ) => match mem_name_lhs.as_str() {
                #[cfg(feature = "cuda")]
                "Cuda" => cuda::cuda_mem_copy(
                    dst,
                    src,
                    size,
                    ffi::cuda::cudaMemcpyKind::cudaMemcpyDeviceToHost,
                )?,
                _ => {
                    error!(
                        "failed to copy data from {} => {}, not implemented",
                        mem_name_lhs, mem_name_rhs
                    );
                    todo!()
                }
            },
            (
                ortn_sys::OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_GPU,
                ortn_sys::OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_GPU,
            ) => match (mem_name_lhs.as_str(), mem_name_rhs.as_str()) {
                #[cfg(feature = "cuda")]
                ("Cuda", "Cuda") => cuda::cuda_mem_copy(
                    dst,
                    src,
                    size,
                    ffi::cuda::cudaMemcpyKind::cudaMemcpyDeviceToDevice,
                )?,
                _ => {
                    error!(
                        "failed to copy data from {} => {}",
                        mem_name_lhs, mem_name_rhs
                    );
                    todo!()
                }
            },
            (_, _) => {
                error!(
                    "failed to copy data from {} => {}",
                    mem_name_lhs, mem_name_rhs
                );
                todo!()
            }
        }
        Ok(())
    }

    fn clone_host(&self) -> Result<ValueAllocated<'static, AllocatorDefault>> {
        let shape_info = self.shape_info()?;
        let mut value =
            ValueAllocated::new(shape_info.dims, &*ALLOCATOR_DEFAULT, shape_info.data_type)?;
        self.assign(&mut value)?;
        Ok(value)
    }

    fn clone_with_allocator<'a, A>(&self, allocator: &'a A) -> Result<ValueAllocated<'a, A>>
    where
        A: AllocatorTrait,
    {
        let shape_info = self.shape_info()?;
        let mut value = ValueAllocated::new(shape_info.dims, allocator, shape_info.data_type)?;
        self.assign(&mut value)?;
        Ok(value)
    }
}

/// `OrtValue` that created from extern data as reference or managed by session
#[derive(Debug)]
pub struct ValueBorrowed<'a, T> {
    pub inner: *mut ffi::OrtValue,
    pub _marker: PhantomData<&'a T>,
}

unsafe impl<'a, T> Sync for ValueBorrowed<'a, T> {}
unsafe impl<'a, T> Send for ValueBorrowed<'a, T> {}

impl<'a, T> ValueTrait for ValueBorrowed<'a, T> {
    fn inner(&self) -> *mut ortn_sys::OrtValue {
        self.inner
    }
}

impl<'a, T> Drop for ValueBorrowed<'a, T> {
    fn drop(&mut self) {
        call_api!(ReleaseValue, self.inner as *mut _);
    }
}

/// helper trait to convert data type like `f32` to `ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT`
pub trait AsONNXTensorElementDataTypeTrait: std::fmt::Debug {
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

/// helper function to convert `ndarray::ArrayView` to `ValueBorrowed`
impl<'a, A, D> TryFrom<ArrayView<'a, A, D>> for ValueBorrowed<'a, ArrayView<'a, A, D>>
where
    A: AsONNXTensorElementDataTypeTrait,
    D: Dimension,
{
    type Error = crate::error::Error;

    fn try_from(value: ArrayView<'a, A, D>) -> std::result::Result<Self, Self::Error> {
        let typ = A::typ();
        let mut mem_info = null_mut();
        rc(call_api!(
            CreateCpuMemoryInfo,
            ffi::OrtAllocatorType::OrtArenaAllocator,
            ffi::OrtMemType::OrtMemTypeDefault,
            &mut mem_info
        ))?;
        let dims = value.shape().iter().map(|d| *d as i64).collect_vec();
        let mut inner = null_mut();
        rc(call_api!(
            CreateTensorWithDataAsOrtValue,
            mem_info,
            value.as_ptr() as *mut _,
            value.len() * size_of::<A>(),
            dims.as_ptr(),
            dims.len(),
            typ,
            &mut inner
        ))?;
        let r = Ok(ValueBorrowed {
            inner,
            _marker: PhantomData,
        });
        trace!(?r, "create");
        r
    }
}

/// `OrtMemoryInfo` trait
pub trait MemoryInfoTrait {
    fn inner(&self) -> *mut ffi::OrtMemoryInfo;

    /// get OrtMemoryInfo name, should be `"Cpu"` or `"Cuda"` or something else
    fn name(&self) -> Result<String> {
        let mut name = null();
        rc(call_api!(MemoryInfoGetName, self.inner(), &mut name))?;
        let name = unsafe { CStr::from_ptr(name) }
            .to_string_lossy()
            .to_string();
        Ok(name)
    }

    /// get OrtMemoryInfo id
    fn id(&self) -> Result<i32> {
        let mut id = 0;
        rc(call_api!(MemoryInfoGetId, self.inner(), &mut id))?;
        Ok(id)
    }

    /// get OrtMemoryInfo mem_type
    fn mem_type(&self) -> Result<ffi::OrtMemType> {
        let mut mem_type = ffi::OrtMemType::OrtMemTypeDefault;
        rc(call_api!(MemoryInfoGetMemType, self.inner(), &mut mem_type))?;
        Ok(mem_type)
    }

    /// get OrtMemoryInfo device_type
    fn device_type(&self) -> Result<ffi::OrtMemoryInfoDeviceType> {
        let mut device_type = ffi::OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_CPU;
        call_api!(MemoryInfoGetDeviceType, self.inner(), &mut device_type);
        Ok(device_type)
    }

    /// check if two OrtMemoryInfo is equal
    fn equal(&self, other: &impl MemoryInfoTrait) -> Result<bool> {
        let mut equal = 0;
        rc(call_api!(
            CompareMemoryInfo,
            self.inner(),
            other.inner(),
            &mut equal
        ))?;
        Ok(equal > 0)
    }
}

/// `OrtMemoryInfo` as `OrtValue` property will drop after `OrtValue` dropped
#[derive(Debug)]
pub struct ConstMemoryInfo {
    pub inner: *const ffi::OrtMemoryInfo,
}

impl MemoryInfoTrait for ConstMemoryInfo {
    fn inner(&self) -> *mut ffi::OrtMemoryInfo {
        self.inner as *mut _
    }
}

/// `OrtMemoryInfo` created by `CreateCpuMemoryInfo` or `CreateMemoryInfo`
#[derive(Debug)]
pub struct MemoryInfo {
    pub inner: *mut ffi::OrtMemoryInfo,
}

impl MemoryInfo {
    pub fn new_cpu() -> Result<Self> {
        let mut inner = null_mut();
        rc(call_api!(
            CreateCpuMemoryInfo,
            ffi::OrtAllocatorType::OrtArenaAllocator,
            ffi::OrtMemType::OrtMemTypeDefault,
            &mut inner
        ))?;
        Ok(MemoryInfo { inner })
    }

    #[cfg(feature = "cuda")]
    pub fn new_cuda(device_id: u32) -> Result<Self> {
        let name = CString::new("Cuda")?;
        let mut inner = null_mut();
        rc(call_api!(
            CreateMemoryInfo,
            name.as_ptr(),
            ffi::OrtAllocatorType::OrtDeviceAllocator,
            device_id as i32,
            ffi::OrtMemType::OrtMemTypeDefault,
            &mut inner
        ))?;
        Ok(MemoryInfo { inner })
    }
}

impl MemoryInfoTrait for MemoryInfo {
    fn inner(&self) -> *mut ffi::OrtMemoryInfo {
        self.inner
    }
}

impl Drop for MemoryInfo {
    fn drop(&mut self) {
        trace!("dropping {:?}", self);
        call_api!(ReleaseMemoryInfo, self.inner);
    }
}

/// `OrtAllocator` trait
pub trait AllocatorTrait: std::fmt::Debug {
    fn inner(&self) -> *mut ffi::OrtAllocator;

    fn free<T>(&self, value: *mut T) {
        call_api!(AllocatorFree, self.inner(), value as *mut c_void);
    }
}

/// `OrtAllocator` created by `GetAllocatorWithDefaultOptions` should not be dropped
#[derive(Debug)]
pub struct AllocatorDefault {
    pub inner: *mut ffi::OrtAllocator,
}

unsafe impl Sync for AllocatorDefault {}
unsafe impl Send for AllocatorDefault {}

impl AllocatorTrait for AllocatorDefault {
    fn inner(&self) -> *mut ffi::OrtAllocator {
        self.inner
    }
}

/// `OrtAllocator` created by `GetAllocatorWithDefaultOptions`
impl AllocatorDefault {
    pub fn new() -> Result<Self> {
        let mut inner = null_mut();
        rc(call_api!(GetAllocatorWithDefaultOptions, &mut inner))?;
        let r = Ok(AllocatorDefault { inner });
        r
    }
}

/// `OrtAllocator` created by `CreateAllocator`
#[derive(Debug)]
pub struct AllocatorSession<'a> {
    pub inner: *mut ffi::OrtAllocator,
    pub session: &'a Session,
}

unsafe impl<'a> Sync for AllocatorSession<'a> {}
unsafe impl<'a> Send for AllocatorSession<'a> {}

impl<'a> AllocatorSession<'a> {
    pub fn new(memory_info: &impl MemoryInfoTrait, session: &'a Session) -> Result<Self> {
        let mut inner = null_mut();
        rc(call_api!(
            CreateAllocator,
            session.inner,
            memory_info.inner(),
            &mut inner
        ))?;
        let r = Ok(AllocatorSession { inner, session });
        trace!(?r, "created");
        r
    }
}

impl Drop for AllocatorSession<'_> {
    fn drop(&mut self) {
        trace!("dropping {:?}", self);
        call_api!(ReleaseAllocator, self.inner);
    }
}

/// `OrtValue` allocated by `OrtAllocator`
#[derive(Debug)]
pub struct ValueAllocated<'a, A>
where
    A: AllocatorTrait,
{
    pub inner: *mut ffi::OrtValue,
    pub _allocator: &'a A,
}

unsafe impl<'a> Sync for ValueAllocated<'a, AllocatorDefault> {}
unsafe impl<'a> Send for ValueAllocated<'a, AllocatorDefault> {}

impl<'a, A> ValueTrait for ValueAllocated<'a, A>
where
    A: AllocatorTrait,
{
    fn inner(&self) -> *mut ffi::OrtValue {
        self.inner
    }
}

impl<'a, A> ValueAllocated<'a, A>
where
    A: AllocatorTrait,
{
    pub fn new(
        dims: impl AsRef<[i64]>,
        allocator: &'a A,
        typ: ONNXTensorElementDataType,
    ) -> Result<Self> {
        let dims = dims.as_ref();
        let mut inner = null_mut();
        rc(call_api!(
            CreateTensorAsOrtValue,
            allocator.inner(),
            dims.as_ptr(),
            dims.len(),
            typ,
            &mut inner
        ))?;
        Ok(ValueAllocated {
            inner,
            _allocator: allocator,
        })
    }
}

impl<'a, A> Drop for ValueAllocated<'a, A>
where
    A: AllocatorTrait,
{
    fn drop(&mut self) {
        trace!("dropping {:?}", self);
        call_api!(ReleaseValue, self.inner as *mut _);
    }
}

#[cfg(feature = "cuda")]
/// cuda api to copy memory between host and device
pub mod cuda {
    use crate::error::*;
    use ortn_sys as ffi;
    use std::ffi::c_void;

    fn rc(code: ffi::cuda::cudaError) -> Result<()> {
        if code != ffi::cuda::cudaError::cudaSuccess {
            return Err(Error::CudaError(code));
        }
        Ok(())
    }

    macro_rules! cuda_call {
        ($func:ident, $a1: expr) => {
            unsafe { ortn_sys::cuda::$func($a1) }
        };

        ($func:ident, $a1: expr, $a2: expr) => {
            unsafe { ortn_sys::cuda::$func($a1, $a2) }
        };

        ($func:ident, $a1: expr, $a2: expr, $a3: expr) => {
            unsafe { ortn_sys::cuda::$func($a1, $a2, $a3) }
        };

        ($func:ident, $a1: expr, $a2: expr, $a3: expr, $a4: expr) => {
            unsafe { ortn_sys::cuda::$func($a1, $a2, $a3, $a4) }
        };
    }

    pub fn set_current_device(device_id: i32) -> Result<()> {
        rc(cuda_call!(cudaSetDevice, device_id))?;
        Ok(())
    }

    pub fn get_current_device() -> Result<i32> {
        let mut device_id = 0;
        rc(cuda_call!(cudaGetDevice, &mut device_id))?;
        Ok(device_id)
    }

    pub fn cuda_mem_copy(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: ffi::cuda::cudaMemcpyKind,
    ) -> Result<()> {
        rc(cuda_call!(cudaMemcpy, dst, src, count, kind))?;
        Ok(())
    }
}
