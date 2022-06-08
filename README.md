# Streamlit Deployment of Multilingual Unsupervised Sentence Simplification

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/johanbekker/mussstreamlit/app.py)

For people who have cognitive disabilities or are not native speakers of a language long and complex sentences can be hard to comprehend. 
For this reason social instances with a large reach, like governmental instances, have to address people in a simple manner. 
For a person who has a thorough understanding of a language, writing a ‘simple’ sentence can be challenging.

This problem is countered in the paper [MUSS: Multilingual Unsupervised Sentence Simplification by Mining Paraphrases](https://github.com/facebookresearch/muss), 
which, as the name suggests, can train a state of the art text simplification model with unlabelled data.

This repository uses the model created in the paper and converts it to an easy to use model format (Pytorch, Hugging Face). It is then deployed using
Streamlit, to show off its capabilities.

## Usage

Clone the repository and install the dependencies in requirements.txt. To download and process the original
model, run prepare_components.py. This will download and convert the models to the default directories.

```
python prepare_components.py
```

To launch the app locally on your system, run the following command while an environment containing the requirements
is active:

```
streamlit run [LOCATION]]/app.py
```

## Speeding up the model

With prepare_components.py, the original fairseq model is downloaded and consequenly converted into a pytorch model
with architecture BartForConditionalGeneration (Hugging Face). The model is saved as a half precision model of approximately 800mb
(Streamlit has a 1GB app size limit).

Further work was done to convert the model to onnx model type, with the idea of model quantization using Hugging Face Optimum, but 
sequence-to-sequence language models were not yet supported for quantization. In future work this is expected to speed up model inference
greatly.

## License

The MUSS license is CC-BY-NC. See the [LICENSE](LICENSE) file for more details.

## Citation

If you use MUSS in your research, please cite [MUSS: Multilingual Unsupervised Sentence Simplification by Mining Paraphrases](https://arxiv.org/abs/2005.00352)

```
@article{martin2021muss,
  title={MUSS: Multilingual Unsupervised Sentence Simplification by Mining Paraphrases},
  author={Martin, Louis and Fan, Angela and de la Clergerie, {\'E}ric and Bordes, Antoine and Sagot, Beno{\^\i}t},
  journal={arXiv preprint arXiv:2005.00352},
  year={2021}
}