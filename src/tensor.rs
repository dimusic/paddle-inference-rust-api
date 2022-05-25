use std::os::raw::{c_ulonglong};

use c_vec::CVec;
use num::{NumCast, ToPrimitive};
use paddle_inference_api_sys::{PD_Tensor, PD_TensorDestroy, PD_TensorReshape, PD_TensorGetShape, PD_OneDimArrayInt32Destroy, PD_TensorSetLod, PD_OneDimArraySize, PD_TwoDimArraySize, PD_TensorGetDataType, PD_DATA_FLOAT32, PD_DATA_UNK, PD_DATA_INT32, PD_DATA_INT64, PD_DATA_INT8, PD_DATA_UINT8, PD_TensorCopyFromCpuInt64, PD_TensorCopyFromCpuFloat, PD_TensorCopyFromCpuUint8, PD_TensorCopyFromCpuInt8, PD_TensorCopyFromCpuInt32, PD_TensorCopyToCpuInt64, PD_TensorGetLod};

#[derive(Debug)]
pub enum PdTensorDataType {
    UNK,
    FLOAT32,
    INT32,
    INT64,
    INT8,
    UINT8,
}

pub trait ToPdTensorDataType {
    fn to_pd_tensor_data_type(&self) -> PdTensorDataType;
}

impl ToPdTensorDataType for f32 {
    fn to_pd_tensor_data_type(&self) -> PdTensorDataType {
        PdTensorDataType::FLOAT32
    }
}

impl ToPdTensorDataType for u8 {
    fn to_pd_tensor_data_type(&self) -> PdTensorDataType {
        PdTensorDataType::UINT8
    }
}

impl ToPdTensorDataType for i8 {
    fn to_pd_tensor_data_type(&self) -> PdTensorDataType {
        PdTensorDataType::INT8
    }
}

impl ToPdTensorDataType for i32 {
    fn to_pd_tensor_data_type(&self) -> PdTensorDataType {
        PdTensorDataType::INT32
    }
}

impl ToPdTensorDataType for i64 {
    fn to_pd_tensor_data_type(&self) -> PdTensorDataType {
        PdTensorDataType::INT64
    }
}

pub trait CopyPdOutput {
    fn copy_pd_output(&mut self, tensor: &PdTensor) -> ();
}

impl CopyPdOutput for Vec<i64> {
    fn copy_pd_output(&mut self, tensor: &PdTensor) -> () {
        println!("copy_pd_output");

        unsafe {
            PD_TensorCopyToCpuInt64(tensor.raw_tensor_ptr, self.as_mut_ptr());
        }
    }
}


fn c_int_arr_to_rust_ints(size: u64, data: *mut i32) -> Vec<i32> {
    let mut res: Vec<i32> = vec![];

    for i in 0..size {
        let item_ptr = unsafe { data.offset(i.try_into().unwrap()) };
        let item = unsafe { *item_ptr };
        res.push(item);
    }

    res
}

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

    /// Reset the shape of the tensor."]
    /// Generally it's only used for the input tensor.
    /// Reshape must be called before calling PD_TensorMutableData*() or
    /// PD_TensorCopyFromCpu*()
    pub fn reshape(&self, shape: &mut Vec<i32>) {
        unsafe {
            let c_vec_shape = CVec::new(shape.as_mut_ptr(), shape.len());
            PD_TensorReshape(self.raw_tensor_ptr, shape.len().try_into().unwrap(), c_vec_shape.into_inner());
        };
    }

    /// Get the tensor shape
    pub fn shape(&self) -> Vec<i32> {
        let shape_ptr = unsafe {
            PD_TensorGetShape(self.raw_tensor_ptr)
        };
        let shape = unsafe { *shape_ptr };
        let input_names_vec = c_int_arr_to_rust_ints(shape.size, shape.data);

        unsafe {
            PD_OneDimArrayInt32Destroy(shape_ptr);
        };

        input_names_vec
    }

    /// Set the tensor lod information
    pub fn set_lod(&self, lod: Vec<Vec<u64>>) {
        let mut pd_arr_lod: Vec<PD_OneDimArraySize> = lod.into_iter().map(|lod_i| {
            let mut converted_lod_data: Vec<c_ulonglong> = lod_i.into_iter().map(|lod_j| {
                lod_j.try_into().unwrap()
            }).collect();
            converted_lod_data.shrink_to_fit();

            let pd_arr_lod = PD_OneDimArraySize {
                size: converted_lod_data.len().try_into().unwrap(),
                data: converted_lod_data.as_mut_ptr()
            };
            std::mem::forget(converted_lod_data);

            pd_arr_lod
        }).collect();

        let mut pd_arr_lod_ptrs: Vec<*mut PD_OneDimArraySize> = pd_arr_lod.iter_mut().map(|pd| {
            let ptr: *mut PD_OneDimArraySize = pd;
            ptr
        }).collect();

        let mut pd_two_dim_arr_lod = PD_TwoDimArraySize {
            size: pd_arr_lod_ptrs.len().try_into().unwrap(),
            data: pd_arr_lod_ptrs.as_mut_ptr()
        };

        std::mem::forget(pd_arr_lod);

        unsafe {
            PD_TensorSetLod(self.raw_tensor_ptr, &mut pd_two_dim_arr_lod);
        };
    }

    /// Get the tensor lod information
    pub fn get_lod(&self) -> Vec<Vec<u64>> {
        let c_lod = unsafe { *PD_TensorGetLod(self.raw_tensor_ptr) };
        let c_lod_size: usize = c_lod.size.try_into().unwrap();
        let c_lod_data = c_lod.data;
        let lod_data = unsafe {
            Vec::from_raw_parts(c_lod_data, c_lod_size, c_lod_size)
        };
        
        lod_data.into_iter().map(|d| {
            let arr = unsafe { *d };
            let size: usize = arr.size.try_into().unwrap();
            let data = arr.data;
            
            unsafe {
                Vec::from_raw_parts(data, size, size)
            }
        }).collect()
    }

    /// Get the tensor data type
    pub fn get_data_type(&self) -> PdTensorDataType {
        let raw_data_type = unsafe {
            PD_TensorGetDataType(self.raw_tensor_ptr)
        };

        match raw_data_type {
            PD_DATA_FLOAT32 => PdTensorDataType::FLOAT32,
            PD_DATA_INT32 => PdTensorDataType::INT32,
            PD_DATA_INT64 => PdTensorDataType::INT64,
            PD_DATA_INT8 => PdTensorDataType::INT8,
            PD_DATA_UINT8 => PdTensorDataType::UINT8,
            PD_DATA_UNK | _ => PdTensorDataType::UNK,
        }
    }

    pub fn get_shape(&self) -> Vec<i32> {
        let shape_ptr = unsafe {
            PD_TensorGetShape(self.raw_tensor_ptr)
        };
        let shape = unsafe { *shape_ptr };

        let converted_shape = c_int_arr_to_rust_ints(shape.size, shape.data);

        unsafe {
            PD_OneDimArrayInt32Destroy(shape_ptr);
        };

        converted_shape
    }

    /// Copy the host memory to tensor data.
    /// It's usually used to set the input tensor data.
    pub fn copy_from_cpu<T>(&self, data: Vec<T>)
    where T: ToPdTensorDataType + ToPrimitive {
        if data.len() == 0 {
            return ();
        }

        let tensor_data_type = data.first().unwrap().to_pd_tensor_data_type();

        match tensor_data_type {
            PdTensorDataType::FLOAT32 => {
                let mut data_typed: Vec<f32> = data.into_iter()
                    .map(|item| NumCast::from(item).unwrap()).collect();

                let data_cvec = unsafe {
                    CVec::<f32>::new(data_typed.as_mut_ptr(), data_typed.len())
                };

                unsafe {
                    PD_TensorCopyFromCpuFloat(self.raw_tensor_ptr, data_cvec.into_inner());
                };
            },
            PdTensorDataType::UINT8 => {
                let mut data_typed: Vec<u8> = data.into_iter()
                    .map(|item| NumCast::from(item).unwrap()).collect();
                
                let data_cvec = unsafe {
                    CVec::<u8>::new(data_typed.as_mut_ptr(), data_typed.len())
                };
                
                unsafe {
                    PD_TensorCopyFromCpuUint8(self.raw_tensor_ptr, data_cvec.into_inner());
                };
            },
            PdTensorDataType::INT8 => {
                let mut data_typed: Vec<i8> = data.into_iter()
                    .map(|item| NumCast::from(item).unwrap()).collect();
                
                let data_cvec = unsafe {
                    CVec::<i8>::new(data_typed.as_mut_ptr(), data_typed.len())
                };

                unsafe {
                    PD_TensorCopyFromCpuInt8(self.raw_tensor_ptr, data_cvec.into_inner());
                };
            },
            PdTensorDataType::INT32 => {
                let mut data_typed: Vec<i32> = data.into_iter()
                    .map(|item| NumCast::from(item).unwrap()).collect();
                
                let data_cvec = unsafe {
                    CVec::<i32>::new(data_typed.as_mut_ptr(), data_typed.len())
                };

                unsafe {
                    PD_TensorCopyFromCpuInt32(self.raw_tensor_ptr, data_cvec.into_inner());
                };
            },
            PdTensorDataType::INT64 => {
                let mut data_typed: Vec<i64> = data.into_iter()
                    .map(|item| NumCast::from(item).unwrap()).collect();
                
                let data_cvec = unsafe {
                    CVec::<i64>::new(data_typed.as_mut_ptr(), data_typed.len())
                };

                unsafe {
                    PD_TensorCopyFromCpuInt64(self.raw_tensor_ptr, data_cvec.into_inner());
                };
            },
            _ => { }
        }
    }

    /// Copy the tensor data to the host memory
    /// It's usually used to get the output tensor data.
    pub fn copy_to_cpu(&self, output_size: usize) -> Vec<i64> {
        let mut output = vec![0; output_size];

        unsafe {
            PD_TensorCopyToCpuInt64(self.raw_tensor_ptr, output.as_mut_ptr());
        };

        output
    }
}

