use paddle_inference_rust_api::{PdConfig, PdPredictor, PdTensor};

// Download models here https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/mobilenetv1.tgz
fn main() {
    let config = PdConfig::new();
    config.disable_gpu();
    config.disable_glog_info();
    config.set_cpu_math_library_num_threads(1);
    config.set_model("./examples/models/mobilenetv1/inference.pdmodel", "./examples/models/mobilenetv1/inference.pdiparams");

    let predictor = PdPredictor::new(&config);

    let input_names = predictor.get_input_names();
    let input_tensor = predictor.get_input_handle(&input_names[0]);

    let output_names = predictor.get_output_names();
    let output_tensor = predictor.get_output_handle(&output_names[0]);

    let data: Vec<f32> = vec![0 as f32; 1*3*224*224].iter().enumerate()
        .map(|(i, _)| {
            let i_float = i as f32;
            (i_float % 255 as f32) * 0.1
        }).collect();

    input_tensor.reshape(vec![1, 3, 244, 244]);
    
    input_tensor.copy_from_cpu(data);

    let lod: Vec<Vec<u64>> = vec![
        vec![10, 0],
        vec![10, 0],
    ];
    println!("LOD {:#?}", lod);
    input_tensor.set_lod(lod);

    predictor.run();

    let output_shape = output_tensor.get_shape();

    println!("output_shape: {:?}", output_shape);
    let mut output_data: Vec<f32> = vec![0 as f32; num_elements(&output_shape)];

    output_tensor.copy_to_cpu(&mut output_data);

    println!("output_data: {:?}", output_data);
    println!("output_lod: {:?}", output_tensor.get_lod());

}

fn num_elements(shape: &[i32]) -> usize {
    let mut n: i32 = 1;

    for i in shape {
        n *= i;
    }

    n.try_into().unwrap()
}
