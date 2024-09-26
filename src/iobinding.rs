use crate::error::{rc, Result};
use crate::macros::call_api;
use crate::session::Session;
use crate::value::{
    AllocatorDefault, AllocatorTrait, MemoryInfo, ValueAllocated, ValueBorrowed, ValueTrait,
    ALLOCATOR_DEFAULT,
};

use std::marker::PhantomData;
use std::ptr::null_mut;

use ortn_sys as ffi;
use tracing::*;

/// `OrtIoBinding` wrapper
#[derive(Debug)]
pub struct IoBinding<'a> {
    pub inner: *mut ffi::OrtIoBinding,
    pub session: &'a Session,
}

unsafe impl<'a> Send for IoBinding<'a> {}
unsafe impl<'a> Sync for IoBinding<'a> {}

impl<'a> IoBinding<'a> {
    /// Create a new `IoBinding` for the given session.
    pub fn new(session: &'a Session) -> Result<Self> {
        let mut inner = null_mut();
        rc(call_api!(CreateIoBinding, session.inner, &mut inner))?;
        Ok(IoBinding { inner, session })
    }

    /// Bind an input tensor to the binding.
    pub fn bind_input(&mut self, index: usize, value: &mut impl ValueTrait) -> Result<()> {
        let input = &self.session.inputs[index];
        rc(call_api!(
            BindInput,
            self.inner,
            input.name.as_ptr(),
            value.inner()
        ))?;
        Ok(())
    }

    /// Bind multiple input tensors to the binding.
    pub fn bind_inputs<T>(&mut self, mut inputs: impl AsMut<[T]>) -> Result<()>
    where
        T: ValueTrait,
    {
        for (i, input) in inputs.as_mut().iter_mut().enumerate() {
            self.bind_input(i, input)?;
        }
        Ok(())
    }

    /// Bind an output tensor to the binding.
    pub fn bind_output(&mut self, index: usize, value: &mut impl ValueTrait) -> Result<()> {
        let output = &self.session.outputs[index];
        rc(call_api!(
            BindOutput,
            self.inner,
            output.name.as_ptr(),
            value.inner()
        ))?;
        Ok(())
    }

    /// Bind multiple output tensors to the binding.
    pub fn bind_outputs<T>(&mut self, mut outputs: impl AsMut<[T]>) -> Result<()>
    where
        T: ValueTrait,
    {
        for (i, output) in outputs.as_mut().iter_mut().enumerate() {
            self.bind_output(i, output)?;
        }
        Ok(())
    }

    /// Bind an output tensor to the device, when output shape is not fixed,
    pub fn bind_output_to_device(&mut self, index: usize, memory_info: &MemoryInfo) -> Result<()> {
        let output = &self.session.outputs[index];
        rc(call_api!(
            BindOutputToDevice,
            self.inner,
            output.name.as_ptr(),
            memory_info.inner
        ))?;
        Ok(())
    }

    /// Bind multiple output tensors to the device, when output shape is not fixed.
    pub fn bind_outputs_to_device(&mut self, memory_info: &MemoryInfo) -> Result<()> {
        for i in 0..self.session.outputs.len() {
            self.bind_output_to_device(i, memory_info)?;
        }
        Ok(())
    }
    /// clear outputs of binding
    pub fn clear_outputs(&mut self) {
        call_api!(ClearBoundOutputs, self.inner)
    }

    /// clear inputs of binding
    pub fn clear_inputs(&mut self) {
        call_api!(ClearBoundInputs, self.inner);
    }

    /// Get the outputs from the binding with default allocator.
    pub fn get_outputs<'b>(
        &'b self,
    ) -> Result<IoBindingOutputs<'b, 'a, 'static, AllocatorDefault>> {
        let mut count = 0;
        let mut inner = null_mut();

        rc(call_api!(
            GetBoundOutputValues,
            self.inner,
            ALLOCATOR_DEFAULT.inner(),
            &mut inner,
            &mut count
        ))?;

        let outputs = (0..count)
            .map(|n| {
                let inner = *unsafe { inner.add(n).as_ref().expect("failed to get value") };
                let value = ValueBorrowed::<IoBinding> {
                    inner,
                    _marker: PhantomData,
                };
                Ok(value)
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(IoBindingOutputs {
            inner,
            values: outputs,
            allocator: &*ALLOCATOR_DEFAULT,
        })
    }
}

#[derive(Debug)]
pub struct IoBindingOutputs<'a, 'b, 'c, A>
where
    A: AllocatorTrait,
{
    pub inner: *mut *mut ffi::OrtValue,
    pub values: Vec<ValueBorrowed<'a, IoBinding<'b>>>,
    pub allocator: &'c A,
}

impl<'a, 'b, 'c, A> IoBindingOutputs<'a, 'b, 'c, A>
where
    A: AllocatorTrait,
{
    pub fn to_vec(&self) -> Result<Vec<ValueAllocated<'static, AllocatorDefault>>> {
        self.values.iter().map(|v| v.clone_host()).collect()
    }

    pub fn values(&self) -> &[ValueBorrowed<'a, IoBinding<'b>>] {
        &self.values
    }
}

impl<'a, 'b, 'c, A> Drop for IoBindingOutputs<'a, 'b, 'c, A>
where
    A: AllocatorTrait,
{
    fn drop(&mut self) {
        trace!(?self, "dropping");
        let values = std::mem::take(&mut self.values);
        drop(values);
        self.allocator.free(self.inner);
    }
}

impl<'a> Drop for IoBinding<'a> {
    fn drop(&mut self) {
        trace!("dropping {:?}", self);
        call_api!(ReleaseIoBinding, self.inner);
    }
}
