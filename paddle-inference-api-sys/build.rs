extern crate bindgen;

use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_path = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let paddle_install_dir = manifest_path.join("Paddle/build/");//PathBuf::from("./Paddle/build/");
    let third_party_install_dir = paddle_install_dir.join("paddle_inference_c_install_dir/third_party/install");
    
    println!("cargo:rerun-if-changed=paddle_wrapper.h");

    println!("cargo:rustc-link-search=native={}", paddle_install_dir.join("paddle_inference_c_install_dir/paddle/lib").display());
    println!("cargo:rustc-link-search=native={}", paddle_install_dir.join("paddle_inference_install_dir/paddle/lib").display());
    println!("cargo:rustc-link-search=native={}", third_party_install_dir.join("cryptopp/lib").display());
    println!("cargo:rustc-link-search=native={}", third_party_install_dir.join("gflags/lib").display());
    println!("cargo:rustc-link-search=native={}", third_party_install_dir.join("glog/lib").display());
    println!("cargo:rustc-link-search=native={}", third_party_install_dir.join("openblas/lib").display());
    println!("cargo:rustc-link-search=native={}", third_party_install_dir.join("protobuf/lib").display());
    println!("cargo:rustc-link-search=native={}", third_party_install_dir.join("utf8proc/lib").display());
    println!("cargo:rustc-link-search=native={}", third_party_install_dir.join("xxhash/lib").display());
    
    // println!("cargo:rustc-link-lib=static=stdc++");
    println!("cargo:rustc-link-lib=dylib=c++");

    println!("cargo:rustc-link-lib=static=cryptopp");
    println!("cargo:rustc-link-lib=static=gflags");
    println!("cargo:rustc-link-lib=static=glog");
    println!("cargo:rustc-link-lib=static=openblas");
    println!("cargo:rustc-link-lib=static=protobuf");
    println!("cargo:rustc-link-lib=static=utf8proc");
    println!("cargo:rustc-link-lib=static=xxhash");

    println!("cargo:rustc-link-lib=static=paddle_inference");
    println!("cargo:rustc-link-lib=static=paddle_inference_c");
    
    let bindings = bindgen::Builder::default()
        .rustfmt_bindings(true)
        .header("./paddle_wrapper.h")
        .detect_include_paths(true)
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("paddle_bindings.rs"))
        .expect("Couldn't write bindings!");
}

fn build_paddle() {
    // build manually for now
    // cd Paddle
    // mkdir build && cd build
    // cmake -DWITH_TESTING=OFF -DWITH_PYTHON=OFF -DWITH_AVX=OFF -DWITH_MKL=OFF -DWITH_GPU=OFF -DON_INFER=ON -DCMAKE_BUILD_TYPE=Release ..
    // make paddle_inference_c -j4

    todo!();

}
