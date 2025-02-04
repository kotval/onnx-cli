use std::str::FromStr;
use std::path::{Path,PathBuf};
use clap::{Parser, ValueEnum};
use image::{ImageBuffer, Luma, Pixel, imageops::FilterType};
use ort::{
    execution_providers::TensorRTExecutionProvider, session::{builder::GraphOptimizationLevel, Session}, tensor::ArrayExtensions, value::TensorRef
};
use ndarray;

#[derive(Clone, Debug, ValueEnum)]
enum OutputTypes{
    Json,
    Human
}
impl FromStr for OutputTypes {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "json" => Ok(OutputTypes::Json),
            "human" => Ok(OutputTypes::Human),
            _ => Err("Invalid value".to_string())
        }
    }
}
#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli{
    #[arg(short, long)]
    image: std::path::PathBuf,
    #[arg(value_enum, default_value = "human")]
    output: OutputTypes
}
const MINST_MODEL_URL: &str = "https://github.com/onnx/models/raw/refs/heads/main/validated/vision/classification/mnist/model/mnist-12.onnx";

fn main() -> ort::Result<()>{
    let LOCAL_MODEL_PATH: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR")).join("model_cache").join("mnist-12.onnx");
    // parse user input
    let args = Cli::parse();
    //load model
    let before_model_load = std::time::Instant::now();
	let model = Session::builder()?
        .with_execution_providers([TensorRTExecutionProvider::default().build()])?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
		.commit_from_url(MINST_MODEL_URL)?;
        //.commit_from_file(&LOCAL_MODEL_PATH)?;
    let model_load_time = before_model_load.elapsed();
    let before_image_load = std::time::Instant::now();
    let input_shape: &Vec<i64> = model.inputs[0].input_type.tensor_dimensions().expect("input0 to be a tensor type");
    // load image, resize to correct size for model
    let image_buffer: ImageBuffer<Luma<u8>, Vec<u8>> = image::open(&args.image)
		.unwrap()
		.resize(input_shape[2] as u32, input_shape[3] as u32, FilterType::Nearest)
		.to_luma8();
    let array: ndarray::Array4::<f32> = ndarray::Array::from_shape_fn((1, 1, 28, 28), |(_, c, j, i)| {
		let pixel = image_buffer.get_pixel(i as u32, j as u32);
		let channels = pixel.channels();
        // normalize range to [0,1]
		(channels[c] as f32) / 255.0
	});
    let image_load_time = before_image_load.elapsed();
    //do inference
    let before_inference = std::time::Instant::now();
	let outputs = model.run(ort::inputs![array]?)?;
    let mut probabilities: Vec<(usize, f32)> = outputs[0]
        .try_extract_tensor()?
        .softmax(ndarray::Axis(1))
        .iter()
        .copied()
        .enumerate()
        .collect::<Vec<_>>();
    // Sort probabilities to find highest
    let inference_time = before_inference.elapsed();
    let total_time = before_model_load.elapsed();
	probabilities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("file: {} prediction: {} probability: {} model_load_time {:.2?}, image_load_time {:.2?} inference_time {:.2?} total_time {:.2?}", &args.image.display(), probabilities[0].0,probabilities[0].1,model_load_time,image_load_time,inference_time,total_time);
    Ok(())
}
