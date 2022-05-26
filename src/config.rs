use std::ffi::CString;

use paddle_inference_api_sys::{PD_ConfigCreate, PD_Config, PD_ConfigDisableGpu, PD_ConfigSetCpuMathLibraryNumThreads, PD_ConfigDisableGlogInfo, PD_ConfigSetModelDir};

#[derive(Debug)]
pub struct PdConfig {
    raw_config_ptr: *mut PD_Config
}

impl Default for PdConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl PdConfig {
    pub fn new() -> Self {
        let raw_config_ptr = unsafe {
            PD_ConfigCreate()
        };

        Self {
            raw_config_ptr
        }
    }

    pub fn get_raw_config_ptr(&self) -> *mut PD_Config {
        self.raw_config_ptr
    }

    pub fn disable_gpu(&self) {
        unsafe {
            PD_ConfigDisableGpu(self.raw_config_ptr);
        }
    }

    pub fn disable_glog_info(&self) {
        unsafe {
            PD_ConfigDisableGlogInfo(self.raw_config_ptr);
        }
    }

    pub fn set_cpu_math_library_num_threads(&self, threads: i32) {
        unsafe {
            PD_ConfigSetCpuMathLibraryNumThreads(self.raw_config_ptr, threads);
        }
    }

    pub fn set_model_dir(&self, model_path: &str) {
        let model_path_c_str = CString::new(model_path).expect("CString failed");

        unsafe {
            PD_ConfigSetModelDir(self.raw_config_ptr, model_path_c_str.as_ptr())
        }
    }
}
