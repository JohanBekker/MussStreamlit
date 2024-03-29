from pathlib import Path
import os
import shutil

# from preprocessing.utils.helpers import log_action
from preprocessing.utils.download_extract import download_and_extract
# from preprocessing.preprocessors import GPT2BPEPreprocessor

from preparation.convert_bart_original_pytorch_checkpoint_to_pytorch import convert_bart_checkpoint
# from preparation.run_onnx_exporter import convert_onnx

from fastBart import export_and_get_onnx_model


MODEL_DIR = Path('models/fairseq/')

def download_extract_model(model_name = "muss_en_wikilarge_mined"):
    model_path = MODEL_DIR / model_name
    if not (model_path / 'model.pt').exists():
        url = f'https://dl.fbaipublicfiles.com/muss/{model_name}.tar.gz'
        model_path.parent.mkdir(parents=True, exist_ok=True)
        download_and_extract(url, model_path)
    return model_path

download_extract_model()
#gp2bpe = GPT2BPEPreprocessor()


fairseq_path = 'models/fairseq/muss_en_wikilarge_mined/model.pt'
pytorch_dump_folder_path = 'models/pytorch_bartmodel/'
hf_config = 'facebook/bart-large'

if not (Path(pytorch_dump_folder_path) / 'pytorch_model.bin').exists():
    Path(pytorch_dump_folder_path).mkdir(parents=True, exist_ok=True)
    # Convert the model from fairseq to pytorch/huggingface
    convert_bart_checkpoint(fairseq_path, pytorch_dump_folder_path, hf_checkpoint_name=hf_config)



model_name = "models/pytorch_bartmodel/"
custom_output_path = "models/onnx_quantized/"

shutil.copy((Path(pytorch_dump_folder_path) / 'config.json'), custom_output_path)

Path(custom_output_path).mkdir(parents=True, exist_ok=True)
if not ".onnx" in ' '.join(os.listdir(custom_output_path)):
    export_and_get_onnx_model(model_name, custom_output_path)



# Run the conversion to onnx filetype. To parameters, navigate to preparation.run_onnx_exporter and change
# the arguments in the parser. To change directories from default, give them to convert_onnx()
# Default onnx_out_path = 'onnx_bart/optimized_bart_onnx.onnx'

# with log_action('Converting torch model to onnx model type'):
#     onnx_dir = 'onnx_bart/'
#     onnx_name = 'bart_onnx.onnx'
#     if not Path(onnx_dir).exists():
#         Path(onnx_dir).mkdir(parents=True, exist_ok=True)
#     if not (Path(onnx_dir) / ('optimized_' + onnx_name)).exists():
#         onnx_out_path = convert_onnx(torch_model_path=pytorch_dump_folder_path, onnx_output_path=None)
#     else:
#         onnx_out_path = Path(onnx_dir) / ('optimized_' + onnx_name)

