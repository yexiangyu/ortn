use std::ffi::{CStr, CString};
use std::ptr::{null, null_mut};
use std::sync::Arc;
use std::time::Instant;

use crate::environment::Environment;
use crate::error::*;
use crate::iobinding::IoBinding;
use crate::macros::call_api;
use crate::value::{AllocatorDefault, ValueBorrowed, ValueTrait};
use itertools::Itertools;
use lazy_static::lazy_static;
use ortn_sys as ffi;
use smart_default::SmartDefault;
use tracing::*;

lazy_static! {
    static ref ENV: Arc<Environment> = Arc::new(
        Environment::builder()
            .build()
            .expect("could not create default environment")
    );
}

#[derive(Debug, Clone)]
pub struct TensorShapeInfo {
    pub name: CString,
    pub dims: Vec<i64>,
    pub data_type: ffi::ONNXTensorElementDataType,
}

#[derive(Debug, Clone)]
pub struct Session {
    pub inner: *mut ffi::OrtSession,
    pub inputs: Vec<TensorShapeInfo>,
    pub outputs: Vec<TensorShapeInfo>,
    _env: Arc<Environment>,
}

unsafe impl Send for Session {}
unsafe impl Sync for Session {}

impl Drop for Session {
    fn drop(&mut self) {
        trace!("dropping {:?}", self);
        call_api!(ReleaseSession, self.inner);
    }
}

impl Session {
    pub fn builder() -> SessionBuilder {
        SessionBuilder::default()
    }

    pub fn run_with_iobinding(&self, io: &mut IoBinding) -> Result<()> {
        let tm = Instant::now();
        rc(call_api!(RunWithBinding, self.inner, null(), io.inner))?;
        trace!("run with binding {:?} delta={:?}", io, tm.elapsed());
        Ok(())
    }

    pub fn run<T>(&self, inputs: impl AsRef<[T]>) -> Result<Vec<ValueBorrowed<Session>>>
    where
        T: ValueTrait,
    {
        let tm = Instant::now();
        let inputs = inputs
            .as_ref()
            .iter()
            .map(|i| i.inner() as *const _)
            .collect_vec();
        let input_names = self.inputs.iter().map(|n| n.name.as_ptr()).collect_vec();
        let mut outputs = self.outputs.iter().map(|_| null_mut()).collect_vec();
        let output_names = self.outputs.iter().map(|n| n.name.as_ptr()).collect_vec();
        rc(call_api!(
            Run,
            self.inner,
            null(),
            input_names.as_ptr(),
            inputs.as_ptr(),
            inputs.len(),
            output_names.as_ptr(),
            output_names.len(),
            outputs.as_mut_ptr()
        ))?;
        let res = Ok(outputs
            .into_iter()
            .map(|inner| ValueBorrowed {
                inner,
                _marker: std::marker::PhantomData,
            })
            .collect_vec());
        trace!(?inputs, ?res, "run, delta={:?}", tm.elapsed());
        res
    }

    fn update_inputs(&mut self) -> Result<()> {
        let mut count = 0;

        rc(call_api!(SessionGetInputCount, self.inner, &mut count))?;

        let allocator = AllocatorDefault::new()?;

        for index in 0..count {
            let mut name_ = null_mut();
            rc(call_api!(
                SessionGetInputName,
                self.inner,
                index,
                allocator.inner,
                &mut name_
            ))?;
            let name = unsafe { CStr::from_ptr(name_) }.to_owned();
            rc(call_api!(AllocatorFree, allocator.inner, name_ as *mut _))?;

            let mut type_ = null_mut();
            rc(call_api!(
                SessionGetInputTypeInfo,
                self.inner,
                index,
                &mut type_
            ))?;

            let mut tensor_type_ = null();

            rc(call_api!(
                CastTypeInfoToTensorInfo,
                type_,
                &mut tensor_type_
            ))?;

            let mut dim_count = 0;

            rc(call_api!(GetDimensionsCount, tensor_type_, &mut dim_count))?;

            let mut dims = vec![0; dim_count];

            rc(call_api!(
                GetDimensions,
                tensor_type_,
                dims.as_mut_ptr(),
                dim_count
            ))?;

            let mut data_type =
                ffi::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;

            rc(call_api!(
                GetTensorElementType,
                tensor_type_,
                &mut data_type
            ))?;

            self.inputs.push(TensorShapeInfo {
                name,
                dims,
                data_type,
            });
        }
        Ok(())
    }

    fn update_outputs(&mut self) -> Result<()> {
        let mut count = 0;
        rc(call_api!(SessionGetOutputCount, self.inner, &mut count))?;
        let allocator = AllocatorDefault::new()?;

        for index in 0..count {
            let mut name_ = null_mut();
            rc(call_api!(
                SessionGetOutputName,
                self.inner,
                index,
                allocator.inner,
                &mut name_
            ))?;

            let name = unsafe { CStr::from_ptr(name_) }.to_owned();

            rc(call_api!(AllocatorFree, allocator.inner, name_ as *mut _))?;

            let mut type_ = null_mut();

            rc(call_api!(
                SessionGetOutputTypeInfo,
                self.inner,
                index,
                &mut type_
            ))?;

            let mut tensor_type_ = null();

            rc(call_api!(
                CastTypeInfoToTensorInfo,
                type_,
                &mut tensor_type_
            ))?;

            let mut dim_count = 0;

            rc(call_api!(GetDimensionsCount, tensor_type_, &mut dim_count))?;

            let mut dims = vec![0; dim_count];

            rc(call_api!(
                GetDimensions,
                tensor_type_,
                dims.as_mut_ptr(),
                dim_count
            ))?;

            let mut data_type =
                ffi::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;

            rc(call_api!(
                GetTensorElementType,
                tensor_type_,
                &mut data_type
            ))?;

            self.outputs.push(TensorShapeInfo {
                name,
                dims,
                data_type,
            });
        }
        Ok(())
    }
}

#[derive(SmartDefault)]
pub struct SessionBuilder {
    env: Option<Arc<Environment>>,
    #[default(1)]
    intra_threads: u16,
    #[default(1)]
    inter_threads: u16,
    use_tensor_rt: bool,
    use_cuda: bool,
    #[default(0)]
    cuda_device_id: i32,
    #[default(0)]
    cuda_mem_limit: usize,
    #[default(ffi::GraphOptimizationLevel::ORT_ENABLE_ALL)]
    graph_optimization_level: ffi::GraphOptimizationLevel,
}

impl SessionBuilder {
    pub fn with_env(mut self, env: impl Into<Arc<Environment>>) -> Self {
        self.env = Some(env.into());
        self
    }

    pub fn with_inter_threads(mut self, threads: u16) -> Self {
        self.inter_threads = threads;
        self
    }

    pub fn with_intra_threads(mut self, threads: u16) -> Self {
        self.intra_threads = threads;
        self
    }

    pub fn with_cuda_device(mut self, device_id: i32) -> Self {
        self.cuda_device_id = device_id;
        self
    }

    pub fn with_cuda_mem_limit(mut self, limit: usize) -> Self {
        self.cuda_mem_limit = limit;
        self
    }

    pub fn with_use_cuda(mut self, use_cuda: bool) -> Self {
        self.use_cuda = use_cuda;
        self
    }

    pub fn with_use_tensor_rt(mut self, use_tensor_rt: bool) -> Self {
        self.use_tensor_rt = use_tensor_rt;
        self
    }

    pub fn with_graph_optimization_level(mut self, level: ffi::GraphOptimizationLevel) -> Self {
        self.graph_optimization_level = level;
        self
    }

    pub fn build(self, model: &[u8]) -> Result<Session> {
        let mut option = null_mut();

        rc(call_api!(CreateSessionOptions, &mut option))?;

        if self.use_cuda {
            let mut cuda_option = null_mut();

            rc(call_api!(CreateCUDAProviderOptions, &mut cuda_option))?;

            rc(call_api!(
                UpdateCUDAProviderOptions,
                cuda_option,
                &CString::new("device_id")?.as_ptr(),
                &CString::new(self.cuda_device_id.to_string())?.as_ptr(),
                1
            ))?;

            if self.cuda_mem_limit > 0 {
                rc(call_api!(
                    UpdateCUDAProviderOptions,
                    cuda_option,
                    &CString::new("gpu_mem_limit")?.as_ptr(),
                    &CString::new(self.cuda_mem_limit.to_string())?.as_ptr(),
                    1
                ))?;
            }

            rc(call_api!(
                SessionOptionsAppendExecutionProvider_CUDA_V2,
                option,
                cuda_option
            ))?;

            call_api!(ReleaseCUDAProviderOptions, cuda_option);
        }

        rc(call_api!(
            SetIntraOpNumThreads,
            option,
            self.intra_threads as i32
        ))?;

        rc(call_api!(
            SetInterOpNumThreads,
            option,
            self.inter_threads as i32
        ))?;

        rc(call_api!(
            SetSessionGraphOptimizationLevel,
            option,
            self.graph_optimization_level
        ))?;

        if self.use_tensor_rt {
            let mut tensor_rt_option = null_mut();
            rc(call_api!(
                CreateTensorRTProviderOptions,
                &mut tensor_rt_option
            ))?;
            rc(call_api!(
                UpdateTensorRTProviderOptions,
                tensor_rt_option,
                &CString::new("device_id")?.as_ptr(),
                &CString::new(self.cuda_device_id.to_string())?.as_ptr(),
                1
            ))?;
            rc(call_api!(
                SessionOptionsAppendExecutionProvider_TensorRT_V2,
                option,
                tensor_rt_option
            ))?;
            call_api!(ReleaseTensorRTProviderOptions, tensor_rt_option);
        }

        let _env = self.env.unwrap_or_else(|| ENV.clone());

        let mut inner = null_mut();

        rc(call_api!(
            CreateSessionFromArray,
            _env.inner(),
            model.as_ptr() as *const _,
            model.len(),
            option,
            &mut inner
        ))?;

        let mut session = Session {
            inner,
            _env,
            inputs: vec![],
            outputs: vec![],
        };

        session.update_inputs()?;
        session.update_outputs()?;

        trace!(?session, "create");

        Ok(session)
    }
}
