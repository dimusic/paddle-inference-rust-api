use paddle_inference_api_sys::{PD_Tensor, PD_TensorDestroy};

#[derive(Debug)]
pub struct PdTensor {
    raw_tensor_ptr: *mut PD_Tensor,
}

impl Drop for PdTensor {
    fn drop(&mut self) {
        unsafe { PD_TensorDestroy(self.raw_tensor_ptr) };
    }
}

impl PdTensor {
    pub fn new(handle: *mut PD_Tensor) -> Self {
        Self {
            raw_tensor_ptr: handle
        }
    }
}
