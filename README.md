# onnx-cli

This cli runs the MNIST model in onnx format on an image using [ort](https://github.com/pykeio/ort).
Supported formats are include anything that [image](https://github.com/image-rs/image/tree/main) can read.
In particular, we support jpg, png, and webp. If you don't already have the onnx for the MNIST model,
it will be downloaded and cached for you on first run.

## Usage
````
Usage: onnx-cli --image <IMAGE> [OUTPUT]

Arguments:
  [OUTPUT]  [default: human] [possible values: json, human]

Options:
  -i, --image <IMAGE>  
  -h, --help           Print help
  -V, --version        Print version
````

## Performance
````
$ cargo run -- -i test_images/7.png 
$> prediction: 7 probability: 0.9999985 model_load_time 20.62ms, image_load_time 397.13µs inference_time 196.13µs total_time 21.21ms
````
The runtime of the model is dominated by the amount of time it takes to load the model into memory. The actual inference is very fast. 
Since we don't support batch processing, we cannot get around loading the model once per call. If we supported batch mode, we would 
only have to pay this price once. 

Tested on MacOS and Archlinux. While ort supports cuda, didn't try the cuda backend. It is not likely that loading the model to the gpu would be
faster than loading it to ram. By modern standards the model is tiny, and it likely wouldn't make much difference to inference time anyway.

