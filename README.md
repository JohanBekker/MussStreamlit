# Streamlit Deployment of Multilingual Unsupervised Sentence Simplification

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/johanbekker/mussstreamlit/app.py)

For people who have cognitive disabilities or are not native speakers of a language long and complex sentences can be hard to comprehend. 
For this reason social instances with a large reach, like governmental instances, have to address people in a simple manner. 
For a person who has a thorough understanding of a language, writing a ‘simple’ sentence can be challenging.

This problem can be solved with the use of deep learning models, where especially the rise of the transformer architecture caused a massive 
improvement in performance in natural language processing (NLP). The problem with deep learning models is the necessity for large quantities of labeled data.

This problem is countered in the paper [MUSS: Multilingual Unsupervised Sentence Simplification by Mining Paraphrases](https://github.com/facebookresearch/muss), 
which, as the name suggests, can train a state of the art text simplification model with unlabelled data. To implement this strategy for 
the Dutch language, I forked the repository. As the author had acces to the Facebook supercomputer cluster, I made the necessary 
alterations to the paraphrase mining code to make it work on my workstation with 32GB RAM and a RTX 2070 super 8GB. The training of the
model is done on an Azure cloud VM using the Azure Python SDK. See Azure_mussNL.ipynb.

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

## Authors

* **Louis Martin** ([louismartincs@gmail.com](mailto:louismartincs@gmail.com))


## Citation

If you use MUSS in your research, please cite [MUSS: Multilingual Unsupervised Sentence Simplification by Mining Paraphrases](https://arxiv.org/abs/2005.00352)

```
@article{martin2021muss,
  title={MUSS: Multilingual Unsupervised Sentence Simplification by Mining Paraphrases},
  author={Martin, Louis and Fan, Angela and de la Clergerie, {\'E}ric and Bordes, Antoine and Sagot, Beno{\^\i}t},
  journal={arXiv preprint arXiv:2005.00352},
  year={2021}
}