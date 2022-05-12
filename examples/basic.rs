use paddle_inference_rust_api::config::PdConfig;

fn main() {
    let config = PdConfig::new();
    config.disable_gpu();
    config.disable_glog_info();
    config.set_cpu_math_library_num_threads(1);
    config.set_model_dir("./lac_models/lac_model/model");
    
    println!("basic example {:?}", config.get_raw_config_ptr());
}
