## Rust bindings for the Paddle Inference API

Rust bindings for the [Paddle Inference library](https://www.paddlepaddle.org.cn/inference/product_introduction/inference_intro.html).

### Requirements
1. **Get paddle_inference_c library.**
You'll need to either [download precompiled binaries](https://paddleinference.paddlepaddle.org.cn/master/user_guides/download_lib.html) or follow directions on the [official docs](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/compile/fromsource.html) to build it from sources.

2. Create env variable **LIB_PADDLE_C_INSTALL_DIR** with the path to your paddle_inference_c lib.

3. Make sure paddle_inference_c shared lib and it's dependencies are added to:
    * **LD_LIBRARY_PATH** for Linux
    * **DYLD_FALLBACK_LIBRARY_PATH** for MacOS
    * **PATH** for Windows

### Example

Use `cargo run --example basic` to run example and check out `examples` folder
