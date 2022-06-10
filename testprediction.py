from transformers import BartForConditionalGeneration, BartTokenizer
# import onnxruntime
# import numpy as np
# from pathlib import Path
# import torch

from preprocessing.preprocessors import get_preprocessors
from preprocessing.preprocessors import ComposedPreprocessor


# def get_muss_preprocessors():
#     language = 'en'
#     preprocessors_kwargs = {
#         'LengthRatioPreprocessor': {'target_ratio': 0.8, 'use_short_name': False},
#         'ReplaceOnlyLevenshteinPreprocessor': {'target_ratio': 0.8, 'use_short_name': False},
#         'WordRankRatioPreprocessor': {'target_ratio': 0.8, 'language': language, 'use_short_name': False},
#         'DependencyTreeDepthRatioPreprocessor': {'target_ratio': 0.8, 'language': language, 'use_short_name': False},
#     }
#     return get_preprocessors(preprocessors_kwargs)

def get_muss_preprocessors():
    language = 'en'
    preprocessors_kwargs = {
        'LengthRatioPreprocessor': {'target_ratio': 0.9, 'use_short_name': False},
        'ReplaceOnlyLevenshteinPreprocessor': {'target_ratio': 0.65, 'use_short_name': False},
        'WordRankRatioPreprocessor': {'target_ratio': 0.75, 'language': language, 'use_short_name': False},
        'DependencyTreeDepthRatioPreprocessor': {'target_ratio': 0.4, 'language': language, 'use_short_name': False},
    }
    return get_preprocessors(preprocessors_kwargs)


preprocessors = get_muss_preprocessors()
composed_preprocessor = ComposedPreprocessor(preprocessors)

# sentence = 'Hello! This is an exquisite example sentence in which I am, exclusively, contemplating utter nonsense. Effective altruism advocates using evidence to determine the most effective ways to benefit others.'
sentence = 'Hello! This is an exquisite example sentence in which I am, exclusively, contemplating utter nonsense.'
# sentence = 'Effective altruism advocates using evidence to determine the most effective ways to benefit others.'

# pytorch_dump_folder_path = 'models/pytorch_bartmodel/'
pytorch_dump_folder_path = 'models/half_precision/'
# pytorch_dump_folder_path = 'torch_quant/'


tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
model = BartForConditionalGeneration.from_pretrained(pytorch_dump_folder_path)
model.eval()


def tokenize_sentence(sentence):
    MAX_SEQUENCE_LENGTH = 1024
    return tokenizer(composed_preprocessor.encode_sentence(sentence), return_tensors="pt",
                     max_length=MAX_SEQUENCE_LENGTH,
                     padding='max_length', add_special_tokens=True)


def clean_output(prediction):
    symbols = ['<s>', '</s>']
    for symbol in symbols:
        prediction = prediction.replace(symbol, '')
    return prediction

inputs = tokenize_sentence(sentence)

summaries = model.generate(**inputs,
                           num_beams=5,
                            max_length=1024,#1024,
                            early_stopping=True,
                            decoder_start_token_id=model.config.decoder_start_token_id)

outp = tokenizer.decode(summaries[0])
cleaned_output = clean_output(outp)
print(cleaned_output)


from fastBart import get_onnx_model
model_onnx = get_onnx_model(model_name="pytorch_bartmodel", onnx_models_path="models/onnx_quantized/")

summaries = model_onnx.generate(**inputs,
                           num_beams=5,
                            max_length=1024,
                            early_stopping=True,
                            decoder_start_token_id=model_onnx.config.decoder_start_token_id)

outp = tokenizer.decode(summaries[0])
cleaned_output = clean_output(outp)
print(cleaned_output)


# x = [2, 0, 2387, 964, 32, 3035, 6, 53, 51, 3529, 350, 171, 33237, 8, 33, 10, 319, 9, 4696, 11, 49, 689, 4, 2]
# y = [2, 2387, 964, 32, 3035, 6, 53, 51, 3529, 350, 171, 33237, 8, 4076, 350, 203, 3766, 4, 2]


# onnx_file_path = 'C:\\Users\\johan\\PycharmProjects\\MussStreamlit\\onnx_bart\\optimized_bart_onnx.onnx'#"onnx_bart/optimized_bart_onnx.onnx"
# onnx_file_path = 'onnx_bart\\optimized_bart_onnx.onnx'
# onnx_file_path = 'onnx_bart\\bart_onnx.onnx'
# onnx_file_path = 'onnx\\model.onnx'
# onnx_file_path = "onnx_optimized\\optimized_bart.onnx"
# onnx_file_path = "onnx_optimized\\bart_onnx.onnx"
# onnx_file_path = "quant\\bart_quantized.onnx"
# num_beams = 5
# max_length = 1024
# decoder_start_token_id = 2

#ort_sess = None
#with torch.no_grad():

# ort_sess = onnxruntime.InferenceSession(onnx_file_path)
# ort_out = ort_sess.run(
#     None,
#     {
#         "input_ids": inputs["input_ids"].cpu().numpy(),
#         "attention_mask": inputs["attention_mask"].cpu().numpy(),
#         # "decoder_input_ids": inputs["input_ids"].cpu().numpy(),
#         # "decoder_attention_mask": inputs["attention_mask"].cpu().numpy(),
#         "num_beams": np.array(num_beams, dtype=np.int64),
#         "max_length": np.array(max_length, dtype=np.int64),
#         "decoder_start_token_id": np.array(decoder_start_token_id, dtype=np.int64),
#     },
# )
#
# print(clean_output(tokenizer.decode(ort_out[0][0])))