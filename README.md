# Streamlit Deployment of Multilingual Unsupervised Sentence Simplification (Dutch)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/johanbekker/mussstreamlit/app.py)

Want to train your own text simplification model? Check out how I did it in my [Github page](https://github.com/JohanBekker/MussStreamlit)

For people who have cognitive disabilities or are not native speakers of a language long and complex sentences can be hard to comprehend. 
For this reason social instances with a large reach, like governmental instances, have to address people in a simple manner. 
For a person who has a thorough understanding of a language, writing a ‘simple’ sentence can be challenging.

This problem is countered in the paper [MUSS: Multilingual Unsupervised Sentence Simplification by Mining Paraphrases](https://github.com/facebookresearch/muss), 
which, as the name suggests, can train a state of the art text simplification model with unlabelled data.

This repository takes the model created by the authors of the paper and converts it to an easy-to-use model format (Pytorch, Hugging Face). It is then deployed using
Streamlit, to show off its capabilities.

## Usage

Clone the repository and install the dependencies in requirements.txt. If you download the supplied Dutch simplification
model (MarianMT), you can use it straight away. If you want to use an English text simplifier instead, code is supplied which
downloads the original model trained by the paper's authors. It is then converted from the Fairseq framework to
PyTorch/HuggingFace and Onnx, after which you can choose which framework to use.

To download and process the original model, run prepare_components.py. This will download and convert the models to the 
default directories.

```
python prepare_components.py
```

To launch the app locally on your system, run the following command while an environment containing the requirements
is active:

```
streamlit run app.py
```

## Docker

To build this app in a Docker image, run the following bash command: 

```bash
docker build -t mussnlstreamlit:latest -f docker/Dockerfile .
```

When the image is built, run it in a container:

```bash
docker run -p 8501:8501 mussnlstreamlit:latest
```

Now you can reach your application in your webbrowser at http://localhost:8501/. Do note that if you change the
application dependencies and/or the directory of the model and tokenizer files, the Dockerfile needs to be changed
accordingly (docker/Dockerfile).

## Speeding up the model

With prepare_components.py, the original fairseq model is downloaded and consequently converted into a PyTorch model
with architecture BartForConditionalGeneration (Hugging Face).

After this the model is converted to the Onnx model type, after which quantization is applied to improve the inference
speed. The Seq2Seq model type is not fully supported though, and to make the conversion to Onnx work the model had to be
split into three different components.

Quantization reduces the model size to 775MB, down from the 1.6GB PyTorch model, but because the Onnx model consists of
three parts, three different Onnxruntime inference sessions have to be created, which is very memory hungry and thus
this quantized model still exceeds the 1GB limit of Streamlit cloud. Locally it runs fine though.

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