from optimum.onnxruntime.configuration import OptimizationConfig
from optimum.onnxruntime import ORTOptimizer

# optimization_config=99 enables all available graph optimisations
optimization_config = OptimizationConfig(optimization_level=99)

pytorch_dump_folder_path = 'pytorch_bartmodel/'
# model_name = model_name = "facebook/bart-large"
optimizer = ORTOptimizer.from_pretrained(
    pytorch_dump_folder_path,
    feature="seq2seq-lm",
)

bart_model_path = "onnx_optimized/bart_onnx.onnx"
optimized_bart_model_path = "onnx_optimized/optimized_bart.onnx"

optimizer.export(
    onnx_model_path=bart_model_path,
    onnx_optimized_model_output_path=optimized_bart_model_path,
    optimization_config=optimization_config,
)