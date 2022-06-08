from transformers import BartForConditionalGeneration, BartTokenizer
import torch
from torch.quantization import quantize_dynamic
from torch import nn
import os

from preprocessing.preprocessors import get_preprocessors
from preprocessing.preprocessors import ComposedPreprocessor


def get_muss_preprocessors():
    language = 'en'
    preprocessors_kwargs = {
        'LengthRatioPreprocessor': {'target_ratio': 0.8, 'use_short_name': False},
        'ReplaceOnlyLevenshteinPreprocessor': {'target_ratio': 0.8, 'use_short_name': False},
        'WordRankRatioPreprocessor': {'target_ratio': 0.8, 'language': language, 'use_short_name': False},
        'DependencyTreeDepthRatioPreprocessor': {'target_ratio': 0.8, 'language': language, 'use_short_name': False},
    }
    return get_preprocessors(preprocessors_kwargs)


preprocessors = get_muss_preprocessors()
composed_preprocessor = ComposedPreprocessor(preprocessors)

#sentence = 'Hi, this is an exquisite example sentence in which I am, exclusively, contemplating nonsense.'
sentence = 'Effective altruism advocates using evidence to determine the most effective ways to benefit others.'


pytorch_dump_folder_path = 'pytorch_bartmodel/'

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
model = BartForConditionalGeneration.from_pretrained(pytorch_dump_folder_path)
model.eval()


model_quantized = quantize_dynamic(
    model=model, qconfig_spec={nn.Linear}, dtype=torch.qint8, inplace=False
)

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

print_size_of_model(model)
print_size_of_model(model_quantized)

def tokenize_sentence(sentence):
    MAX_SEQUENCE_LENGTH = 1024
    return tokenizer(composed_preprocessor.encode_sentence(sentence), return_tensors="pt",
                     max_length=MAX_SEQUENCE_LENGTH,
                     padding='max_length', add_special_tokens=False)


def clean_output(prediction):
    symbols = ['<s>', '</s>']
    for symbol in symbols:
        prediction = prediction.replace(symbol, '')
    return prediction

inputs = tokenize_sentence(sentence)

dummy_inputs = (torch.randint(1, 10000, (1,1024)), torch.ones(1024).unsqueeze(0))

# traced_model = torch.jit.trace(model_quantized, (inputs.data['input_ids'], inputs.data['attention_mask']))
traced_model = torch.jit.trace(model_quantized, dummy_inputs)
torch.jit.save(traced_model, "torch_quant/bart_traced_eager_quant.pt")

# quant_folder = "torch_quant/"
# model_quantized.save_pretrained(quant_folder)