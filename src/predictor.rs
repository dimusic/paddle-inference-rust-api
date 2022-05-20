use std::ffi::{CString, CStr};

use paddle_inference_api_sys::{PD_Predictor, PD_PredictorCreate, PD_PredictorDestroy, PD_PredictorGetInputNames, PD_OneDimArrayCstrDestroy};

use crate::config::PdConfig;

fn c_str_arr_to_rust_string(size: u64, data: *mut *mut i8) -> Vec<String> {
    let mut res: Vec<String> = vec![];

    for i in 0..size {
        let item_ptr = unsafe { data.offset(i.try_into().unwrap()) };
        let cstr = unsafe { CStr::from_ptr(*item_ptr) };
        let rust_str = cstr.to_str().unwrap().to_owned();
        
        res.push(rust_str);
    }

    res
}

#[derive(Debug)]
pub struct PdPredictor {
    raw_predictor_ptr: *mut PD_Predictor,
}

impl Drop for PdPredictor {
    fn drop(&mut self) {
        unsafe { PD_PredictorDestroy(self.raw_predictor_ptr) };
    }
}

impl PdPredictor {
    pub fn new(config: &PdConfig) -> Self {
        let raw_predictor_ptr = unsafe {
            PD_PredictorCreate(config.get_raw_config_ptr())
        };

        Self {
            raw_predictor_ptr: raw_predictor_ptr
        }
    }

    pub fn get_raw_config_ptr(&self) -> *mut PD_Predictor {
        self.raw_predictor_ptr
    }

    pub fn get_input_names(&self) -> Vec<String> {
        let input_names_ptr = unsafe {
            PD_PredictorGetInputNames(self.raw_predictor_ptr)
        };
        let input_names = unsafe { *input_names_ptr };

        let names_count = input_names.size;
        let names_arr = input_names.data;

        let input_names_vec = c_str_arr_to_rust_string(names_count, names_arr);

        unsafe {
            PD_OneDimArrayCstrDestroy(input_names_ptr);
        };

        input_names_vec
    }
}
