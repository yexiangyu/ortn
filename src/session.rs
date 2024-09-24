use std::ffi::{CStr, CString};
use std::ptr::{null, null_mut};
use std::sync::Arc;

use crate::api::API;
use crate::environment::Environment;
use crate::error::*;
use crate::value::{ValueOutput, ValueTrait};
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

#[derive(Debug)]
pub struct TensorShapeInfo {
    pub name: CString,
    pub dims: Vec<i64>,
    pub data_type: ffi::ONNXTensorElementDataType,
}

#[derive(Debug)]
pub struct Session {
    inner: *mut ffi::OrtSession,
    pub inputs: Vec<TensorShapeInfo>,
    pub outputs: Vec<TensorShapeInfo>,
    _env: Arc<Environment>,
}

unsafe impl Send for Session {}
unsafe impl Sync for Session {}

impl Drop for Session {
    fn drop(&mut self) {
        trace!("dropping {:?}", self);
        unsafe {
            API.ReleaseSession
                .as_ref()
                .expect("failed to get ReleaseSession")(self.inner)
        };
    }
}

impl Session {
    pub fn builder() -> SessionBuilder {
        SessionBuilder::default()
    }
    
    pub fn run<T>(&self, inputs: impl AsRef<[T]>) -> Result<Vec<ValueOutput>>
    where
        T: ValueTrait,
    {
        let inputs = inputs
            .as_ref()
            .iter()
            .map(|i| i.inner() as *const _)
            .collect_vec();
        let input_names = self.inputs.iter().map(|n| n.name.as_ptr()).collect_vec();
        let mut outputs = self.outputs.iter().map(|_| null_mut()).collect_vec();
        let output_names = self.outputs.iter().map(|n| n.name.as_ptr()).collect_vec();
        rc(unsafe {
            API.Run.as_ref().expect("failed to get Run")(
                self.inner,
                null(),
                input_names.as_ptr(),
                inputs.as_ptr(),
                inputs.len(),
                output_names.as_ptr(),
                output_names.len(),
                outputs.as_mut_ptr(),
            )
        })?;
        Ok(outputs
            .into_iter()
            .map(|inner| ValueOutput::new(inner, self))
            .collect_vec())
    }

    fn update_inputs(&mut self) -> Result<()> {
        let mut count = 0;
        rc(unsafe {
            API.SessionGetInputCount
                .as_ref()
                .expect("failed to get SessionGetInputCount")(self.inner, &mut count)
        })?;

        let mut allocator = null_mut();

        rc(unsafe {
            API.GetAllocatorWithDefaultOptions
                .as_ref()
                .expect("failed to get GetAllocatorWithDefaultOptions")(&mut allocator)
        })?;

        for index in 0..count {
            let mut name_ = null_mut();
            rc(unsafe {
                API.SessionGetInputName
                    .as_ref()
                    .expect("failed to get SessionGetInputName")(
                    self.inner, index, allocator, &mut name_,
                )
            })?;

            let name = unsafe { CStr::from_ptr(name_) }.to_owned();

            rc(unsafe {
                API.AllocatorFree
                    .as_ref()
                    .expect("failed to get AllocatorFree")(
                    allocator, name_ as *mut _
                )
            })?;

            let mut type_ = null_mut();

            rc(unsafe {
                API.SessionGetInputTypeInfo
                    .as_ref()
                    .expect("failed to get SessionGetInputTypeInfo")(
                    self.inner, index, &mut type_
                )
            })?;

            let mut tensor_type_ = null();

            rc(unsafe {
                API.CastTypeInfoToTensorInfo
                    .as_ref()
                    .expect("failed to get CastTypeInfoToTensorInfo")(
                    type_, &mut tensor_type_
                )
            })?;

            let mut dim_count = 0;

            rc(unsafe {
                API.GetDimensionsCount
                    .as_ref()
                    .expect("failed to get GetTensorShapeElementCount")(
                    tensor_type_,
                    &mut dim_count,
                )
            })?;

            let mut dims = vec![0; dim_count];

            rc(unsafe {
                API.GetDimensions
                    .as_ref()
                    .expect("failed to get GetDimensions")(
                    tensor_type_,
                    dims.as_mut_ptr(),
                    dim_count,
                )
            })?;

            let mut data_type =
                ffi::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;

            rc(unsafe {
                API.GetTensorElementType
                    .as_ref()
                    .expect("failed to get GetTensorElementType")(
                    tensor_type_, &mut data_type
                )
            })?;

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
        rc(unsafe {
            API.SessionGetOutputCount
                .as_ref()
                .expect("failed to get SessionGetOutputCount")(self.inner, &mut count)
        })?;

        let mut allocator = null_mut();

        rc(unsafe {
            API.GetAllocatorWithDefaultOptions
                .as_ref()
                .expect("failed to get GetAllocatorWithDefaultOptions")(&mut allocator)
        })?;

        for index in 0..count {
            let mut name_ = null_mut();
            rc(unsafe {
                API.SessionGetOutputName
                    .as_ref()
                    .expect("failed to get SessionGetOutputName")(
                    self.inner, index, allocator, &mut name_,
                )
            })?;

            let name = unsafe { CStr::from_ptr(name_) }.to_owned();

            rc(unsafe {
                API.AllocatorFree
                    .as_ref()
                    .expect("failed to get AllocatorFree")(
                    allocator, name_ as *mut _
                )
            })?;

            let mut type_ = null_mut();

            rc(unsafe {
                API.SessionGetOutputTypeInfo
                    .as_ref()
                    .expect("failed to get SessionGetOutputTypeInfo")(
                    self.inner, index, &mut type_
                )
            })?;

            let mut tensor_type_ = null();

            rc(unsafe {
                API.CastTypeInfoToTensorInfo
                    .as_ref()
                    .expect("failed to get CastTypeInfoToTensorInfo")(
                    type_, &mut tensor_type_
                )
            })?;

            let mut dim_count = 0;

            rc(unsafe {
                API.GetDimensionsCount
                    .as_ref()
                    .expect("failed to get GetTensorShapeElementCount")(
                    tensor_type_,
                    &mut dim_count,
                )
            })?;

            let mut dims = vec![0; dim_count];

            rc(unsafe {
                API.GetDimensions
                    .as_ref()
                    .expect("failed to get GetDimensions")(
                    tensor_type_,
                    dims.as_mut_ptr(),
                    dim_count,
                )
            })?;

            let mut data_type =
                ffi::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;

            rc(unsafe {
                API.GetTensorElementType
                    .as_ref()
                    .expect("failed to get GetTensorElementType")(
                    tensor_type_, &mut data_type
                )
            })?;

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

    pub fn with_cuda_deivice(mut self, device_id: i32) -> Self {
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

    pub fn with_graph_optimization_level(
        mut self,
        level: ffi::GraphOptimizationLevel,
    ) -> Self {
        self.graph_optimization_level = level;
        self
    }

    pub fn build(self, model: &[u8]) -> Result<Session> {
        let mut option = null_mut();
        rc(unsafe {
            API.CreateSessionOptions
                .as_ref()
                .expect("failed to get CreateSessionOptions")(&mut option)
        })?;

        if self.use_cuda {

            let mut cuda_option = null_mut();

            rc(unsafe {
                API.CreateCUDAProviderOptions
                    .as_ref()
                    .expect("failed to get CreateCUDAProviderOptions")(
                    &mut cuda_option
                )
            })?;


            rc(unsafe {
                API.UpdateCUDAProviderOptions
                    .as_ref()
                    .expect("failed to get UpdateCUDAProviderOptionsWithValue")(
                    cuda_option,
                    &CString::new("device_id")?.as_ptr(),
                    &CString::new(self.cuda_device_id.to_string())?.as_ptr(),
                    1,
                )
            })?;

            if self.cuda_mem_limit > 0 {
                rc(unsafe {
                    API.UpdateCUDAProviderOptions
                        .as_ref()
                        .expect("failed to get UpdateCUDAProviderOptionsWithValue")(
                        cuda_option,
                        &CString::new("gpu_mem_limit")?.as_ptr(),
                        &CString::new(self.cuda_mem_limit.to_string())?.as_ptr(),
                        1,
                    )
                })?;
            }

            rc(unsafe {
                API.SessionOptionsAppendExecutionProvider_CUDA_V2
                    .as_ref()
                    .expect("failed to get SessionOptionsAppendExecutionProvider_CUDA_V2")(
                    option,
                    cuda_option,
                )
            })?;
        }

        rc(unsafe {
            API.SetIntraOpNumThreads
                .as_ref()
                .expect("failed to get SetIntraOpNumThreads")(
                option, self.intra_threads as i32
            )
        })?;

        rc(unsafe {
            API.SetInterOpNumThreads
                .as_ref()
                .expect("failed to get SetInterOpNumThreads")(
                option, self.inter_threads as i32
            )
        })?;

        rc(unsafe {
            API.SetSessionGraphOptimizationLevel
                .as_ref()
                .expect("failed to get SetSessionGraphOptimizationLevel")(
                option, self.graph_optimization_level
            )
        })?;

        if self.use_tensor_rt
        {
            let mut tensor_rt_option = null_mut();

            rc(unsafe {
                API.CreateTensorRTProviderOptions
                    .as_ref()
                    .expect("failed to get CreateTensorRTProviderOptions")(
                    &mut tensor_rt_option
                )
            })?;

            rc(unsafe {
                API.UpdateTensorRTProviderOptions
                    .as_ref()
                    .expect("failed to get UpdateTensorRTProviderOptions")(
                    tensor_rt_option,
                    &CString::new("device_id")?.as_ptr(),
                    &CString::new(self.cuda_device_id.to_string())?.as_ptr(),
                    1,
                )
            })?;
        }

        let _env = self.env.unwrap_or_else(|| ENV.clone());

        let mut inner = null_mut();

        rc(unsafe {
            API.CreateSessionFromArray
                .as_ref()
                .expect("failed to get CreateSession")(
                _env.inner(),
                model.as_ptr() as *const _,
                model.len(),
                option,
                &mut inner,
            )
        })?;

        let mut session = Session {
            inner,
            _env,
            inputs: vec![],
            outputs: vec![],
        };

        session.update_inputs()?;
        session.update_outputs()?;

        Ok(session)
    }
}
