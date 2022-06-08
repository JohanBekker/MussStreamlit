#from onnxruntime_tools import optimizer
# from onnxruntime_tools.transformers import BartOnnxModel
# from onnxruntime.transformers import optimizer
#
# input_model_path = "onnx_bart\\bart_onnx.onnx"
# export_model_path = "onnx_optimized/"
#
# opt_model = optimizer.optimize_model(
#     input_model_path,
#     'bert',
#     num_heads=16,
#     hidden_size=1024)
#
# opt_model.save_model_to_file('bert.opt.onnx')

from optimum.onnxruntime.configuration import AutoQuantizationConfig
from optimum.onnxruntime import ORTQuantizer

# The model we wish to quantize
# model_checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
pytorch_dump_folder_path = 'pytorch_bartmodel/'
# The type of quantization to apply
qconfig = AutoQuantizationConfig.arm64(is_static=False, per_channel=False)
quantizer = ORTQuantizer.from_pretrained(pytorch_dump_folder_path, feature="seq2seq-lm")

# Quantize the model!
quantizer.export(
    onnx_model_path="quant/bart_model.onnx",
    onnx_quantized_model_output_path="quant/bart_quantized.onnx",
    quantization_config=qconfig,
)