use crate::error::{rc, Result};
use crate::macros::call_api;
use crate::session::Session;
use crate::value::{AllocatorTrait, MemoryInfo, ValueAllocated, ValueTrait};
use std::ptr::null_mut;

use itertools::Itertools;
use ortn_sys as ffi;
use tracing::trace;

// use crate::api::API;

#[derive(Debug)]
pub struct IoBinding<'a> {
    pub inner: *mut ffi::OrtIoBinding,
    pub session: &'a Session,
}

impl<'a> IoBinding<'a> {
    pub fn new(session: &'a Session) -> Result<Self> {
        let mut inner = null_mut();
        rc(call_api!(CreateIoBinding, session.inner, &mut inner))?;
        Ok(IoBinding { inner, session })
    }

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

    pub fn bind_inputs<T>(&mut self, mut inputs: impl AsMut<[T]>) -> Result<()>
    where
        T: ValueTrait,
    {
        for (i, input) in inputs.as_mut().iter_mut().enumerate() {
            self.bind_input(i, input)?;
        }
        Ok(())
    }

    pub fn bind_output(&mut self, index: usize, value: &mut impl ValueTrait) -> Result<()> {
        let output = &self.session.outputs[index];
        rc(call_api!(BindOutput, self.inner, output.name.as_ptr(), value.inner()))?;
        Ok(())
    }

    pub fn bind_outputs<T>(&mut self, mut outputs: impl AsMut<[T]>) -> Result<()>
    where
        T: ValueTrait,
    {
        for (i, output) in outputs.as_mut().iter_mut().enumerate() {
            self.bind_output(i, output)?;
        }
        Ok(())
    }

    pub fn bind_output_to_device(&mut self, index: usize, memory_info: &MemoryInfo) -> Result<()> {
        let output = &self.session.outputs[index];
        rc(call_api!(BindOutputToDevice, self.inner, output.name.as_ptr(), memory_info.inner))?;
        Ok(())
    }

    pub fn bind_outputs_to_device(&mut self, memory_info: &MemoryInfo) -> Result<()> {
        for i in 0..self.session.outputs.len() {
            self.bind_output_to_device(i, memory_info)?;
        }
        Ok(())
    }

    pub fn outputs<A>(&self, allocator: &'a A) -> Result<Vec<ValueAllocated<A>>>
    where
        A: AllocatorTrait,
    {
        let mut count = 0;
        let mut values = null_mut();
        rc(call_api!(GetBoundOutputValues, self.inner, allocator.inner(), &mut values, &mut count))?;

        Ok((0..count)
            .map(|n| {
                let inner = *unsafe { values.add(n).as_ref().expect("failed to get value") };
                ValueAllocated { inner, allocator }
            })
            .collect_vec())
    }
}

impl<'a> Drop for IoBinding<'a> {
    fn drop(&mut self) {
        trace!("dropping {:?}", self);
        call_api!(ReleaseIoBinding, self.inner);
    }
}
