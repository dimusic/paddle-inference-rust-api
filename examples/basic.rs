use paddle_inference_rust_api::{PdConfig, PdPredictor};

fn main() {
    let config = PdConfig::new();
    config.disable_gpu();
    config.disable_glog_info();
    config.set_cpu_math_library_num_threads(1);
    config.set_model_dir("./examples/models_general/lac_model/model");
    println!("config {:?}", config.get_raw_config_ptr());

    let predictor = PdPredictor::new(&config);
    println!("predictor {:?}", predictor.get_raw_predictor_ptr());

    let input_names = predictor.get_input_names();
    println!("input_names: {:?}", input_names);

    let input_handle = predictor.get_input_handle(&input_names[0]);
    println!("input_handle: {:?}", input_handle);

    let output_names = predictor.get_output_names();
    println!("output_names: {:?}", output_names);

    let output_handle = predictor.get_output_handle(&input_names[0]);
    println!("output_handle: {:?}", output_handle);

}
