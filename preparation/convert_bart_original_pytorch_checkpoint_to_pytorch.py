# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert BART checkpoint."""

# https://github.com/huggingface/transformers/blob/main/src/transformers/models/bart/convert_bart_original_pytorch_checkpoint_to_pytorch.py

import argparse
import os
from pathlib import Path
import numpy as np

import fairseq
import torch
from packaging import version
from torch import nn

from transformers import (
    BartConfig,
    BartForConditionalGeneration,
    BartForSequenceClassification,
    BartModel,
    BartTokenizer,
)
from transformers.utils import logging

FAIRSEQ_MODELS = ["bart.large", "bart.large.mnli", "bart.large.cnn", "bart_xsum/model.pt"]
extra_arch = {"bart.large": BartModel, "bart.large.mnli": BartForSequenceClassification}
if version.parse(fairseq.__version__) < version.parse("0.9.0"):
    raise Exception("requires fairseq >= 0.9.0")

logging.set_verbosity_info()
logger = logging.get_logger(__name__)

SAMPLE_TEXT = " Hello world! cécé herlolip"

mnli_rename_keys = [
    ("model.classification_heads.mnli.dense.weight", "classification_head.dense.weight"),
    ("model.classification_heads.mnli.dense.bias", "classification_head.dense.bias"),
    ("model.classification_heads.mnli.out_proj.weight", "classification_head.out_proj.weight"),
    ("model.classification_heads.mnli.out_proj.bias", "classification_head.out_proj.bias"),
]


def remove_ignore_keys_(state_dict):
    ignore_keys = [
        "encoder.version",
        "decoder.version",
        "model.encoder.version",
        "model.decoder.version",
        "_float_tensor",
        "decoder.output_projection.weight"  # Added this layer to the ignore list
    ]
    for k in ignore_keys:
        state_dict.pop(k, None)


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


def load_xsum_checkpoint(checkpoint_path):
    """Checkpoint path should end in model.pt"""
    sd = torch.load(checkpoint_path, map_location="cpu")
    hub_interface = torch.hub.load("pytorch/fairseq", "bart.large.cnn").eval()
    hub_interface.model.load_state_dict(sd["model"])
    return hub_interface


def make_linear_from_emb(emb):
    vocab_size, emb_size = emb.weight.shape
    lin_layer = nn.Linear(vocab_size, emb_size, bias=False)
    lin_layer.weight.data = emb.weight.data
    return lin_layer


def add_zeros_to_embedding(embedding):
    """
    Fairseq model removed the embedding for the <mask> token, which causes an embedding dimensionality mismatch.
    To fix the dimensionality of the finetuned Muss model to match the standard config, a row of zeroes is appended
    to the embedding matrix. The mask token will never be used anyways.
    """
    ch = embedding.tolist()
    ch.append(np.zeros(len(ch[0])))
    return torch.tensor(np.array(ch), dtype=torch.float32)


@torch.no_grad()
def convert_bart_checkpoint(checkpoint_path, pytorch_dump_folder_path, hf_checkpoint_name=None):
    """
    Copy/paste/tweak model's weights to our BERT structure.
    """
    if not os.path.exists(checkpoint_path):
        bart = torch.hub.load("pytorch/fairseq", checkpoint_path).eval()
    else:
        bart = load_xsum_checkpoint(checkpoint_path)

    bart.model.upgrade_state_dict(bart.model.state_dict())
    if hf_checkpoint_name is None:
        hf_checkpoint_name = checkpoint_path.replace(".", "-")
    config = BartConfig.from_pretrained(hf_checkpoint_name)
    tokens = bart.encode(SAMPLE_TEXT).unsqueeze(0)
    tokens2 = BartTokenizer.from_pretrained(hf_checkpoint_name).encode(SAMPLE_TEXT, return_tensors="pt").unsqueeze(0)
    assert torch.eq(tokens, tokens2).all()

    if checkpoint_path == "bart.large.mnli":
        state_dict = bart.state_dict()
        remove_ignore_keys_(state_dict)
        state_dict["model.shared.weight"] = state_dict["model.decoder.embed_tokens.weight"]
        for src, dest in mnli_rename_keys:
            rename_key(state_dict, src, dest)
        model = BartForSequenceClassification(config).eval()
        model.load_state_dict(state_dict)
        fairseq_output = bart.predict("mnli", tokens, return_logits=True)
        new_model_outputs = model(tokens)[0]  # logits
    else:  # no classification heads to worry about
        state_dict = bart.model.state_dict()
        # Add zeroes row to embedding matrix to fix dimensionality mismatch.
        state_dict['encoder.embed_tokens.weight'] = add_zeros_to_embedding(state_dict['encoder.embed_tokens.weight'])
        state_dict['decoder.embed_tokens.weight'] = add_zeros_to_embedding(state_dict['decoder.embed_tokens.weight'])

        remove_ignore_keys_(state_dict)
        state_dict["shared.weight"] = state_dict["decoder.embed_tokens.weight"]
        fairseq_output = bart.extract_features(tokens)
        # if hf_checkpoint_name == "facebook/bart-large":
        #     model = BartModel(config).eval()
        #     model.load_state_dict(state_dict)
        #     new_model_outputs = model(tokens)[0]
        # else:
        model = BartForConditionalGeneration(config).eval()  # an existing summarization ckpt
        model.model.load_state_dict(state_dict)
        if hasattr(model, "lm_head"):
            model.lm_head = make_linear_from_emb(model.model.shared)
        new_model_outputs = model.model(tokens)[0]

    # Check results
    assert fairseq_output.shape == new_model_outputs.shape
    assert (fairseq_output == new_model_outputs).all().item()
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    model.save_pretrained(pytorch_dump_folder_path)

# fairseq_path = 'model/muss_en_wikilarge_mined/model.pt'
# pytorch_dump_folder_path = 'pytorch_bartmodel/'
# hf_config = 'facebook/bart-large'
#
# convert_bart_checkpoint(fairseq_path, pytorch_dump_folder_path, hf_checkpoint_name=hf_config)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     # Required parameters
#     parser.add_argument(
#         "fairseq_path", type=str, help="bart.large, bart.large.cnn or a path to a model.pt on local filesystem."
#     )
#     parser.add_argument("pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
#     parser.add_argument(
#         "--hf_config", default=None, type=str, help="Which huggingface architecture to use: bart-large-xsum"
#     )
#     args = parser.parse_args()
#     convert_bart_checkpoint(args.fairseq_path, args.pytorch_dump_folder_path, hf_checkpoint_name=args.hf_config)
