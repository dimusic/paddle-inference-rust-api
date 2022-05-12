use paddle_inference_api_sys::PD_Predictor;

pub struct PdPredictor {
    raw_predictor_ptr: *mut PD_Predictor,
}
