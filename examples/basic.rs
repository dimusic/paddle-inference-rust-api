use paddle_inference_rust_api::{config::PdConfig, predictor::PdPredictor};

fn main() {
    let config = PdConfig::new();
    config.disable_gpu();
    config.disable_glog_info();
    config.set_cpu_math_library_num_threads(1);
    config.set_model_dir("./examples/models_general/lac_model/model");
    println!("config {:?}", config.get_raw_config_ptr());

    let predictor = PdPredictor::new(&config);
    println!("predictor {:?}", predictor.get_raw_config_ptr());

    let input_names = predictor.get_input_names();
    println!("input_names: {:?}", input_names);

}
