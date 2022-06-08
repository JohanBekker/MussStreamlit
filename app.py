import streamlit as st
import webbrowser

from transformers import BartForConditionalGeneration, BartTokenizer

from preprocessing.preprocessors import get_preprocessors
from preprocessing.preprocessors import ComposedPreprocessor

from torch.nn.parameter import Parameter

st.set_page_config(page_title='Text simplifier', layout="wide")


def get_muss_preprocessors(length=0.8, replace=0.8, word=0.8, tree=0.8):
    language = 'en'
    preprocessors_kwargs = {
        'LengthRatioPreprocessor': {'target_ratio': length, 'use_short_name': False},
        'ReplaceOnlyLevenshteinPreprocessor': {'target_ratio': replace, 'use_short_name': False},
        'WordRankRatioPreprocessor': {'target_ratio': word, 'language': language, 'use_short_name': False},
        'DependencyTreeDepthRatioPreprocessor': {'target_ratio': tree, 'language': language, 'use_short_name': False},
    }
    return get_preprocessors(preprocessors_kwargs)


@st.cache(show_spinner=True, hash_funcs={Parameter: lambda _: None})
def load_model():
    model_path = 'models/half_precision/'
    hf_model = BartForConditionalGeneration.from_pretrained(model_path)
    hf_model.eval()
    return hf_model


def clean_output(prediction):
    symbols = ['<s>', '</s>']
    for symbol in symbols:
        prediction = prediction.replace(symbol, '')
    return prediction


model = load_model()
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')


def tokenize_sentence(sentence, MAX_SEQUENCE_LENGTH=1024):
    return tokenizer(composed_preprocessor.encode_sentence(sentence), return_tensors="pt",
                     max_length=MAX_SEQUENCE_LENGTH,
                     padding='max_length', add_special_tokens=False)


def simplify(sentence):
    tokens = tokenize_sentence(sentence)
    tokenized_simplification = model.generate(**tokens, num_beams=4,
                                              # max_length=(sum(tokens.attention_mask[0])//1.25), #early_stopping=False,
                                              decoder_start_token_id=model.config.decoder_start_token_id)
    return clean_output(tokenizer.decode(tokenized_simplification[0]))


def open_url(url):
    webbrowser.open_new_tab(url)


st.title("English Text Simplifier")
st.subheader("Enter the sentence(s) to be simplified and we'll take care of the rest!")
st.markdown("***")

m = st.markdown(
    """ <style> div.stButton > button:first-child { background-color: black; width: 18em; color: white} </style>""",
    unsafe_allow_html=True)

st.sidebar.markdown("<h1 style='text-align: center; color: black;'>Check me out!</h1>", unsafe_allow_html=True)
if st.sidebar.button("Portfolio        "):
    open_url('https://www.datascienceportfol.io/JohanBekker')
if st.sidebar.button("GitHub        "):
    open_url('https://github.com/JohanBekker')
if st.sidebar.button("LinkedIn        "):
    open_url('https://www.linkedin.com/in/johan-bekker-3501a6168/')

c = st.container()
c2 = st.container()

col1, col2, col3 = c2.columns([3, 1, 1])
col1.subheader('Play around with the parameters and see the results!')
col1.write("More information about the parameters can be found in the reference below.")
length_ratio = col1.slider("Length ratio:", value=0.8)
replace_ratio = col1.slider("Levenshtein replace ratio:", value=0.8)
word_ratio = col1.slider("Wordrank ratio:", value=0.8)
treedepth_ratio = col1.slider("Dependency tree depth ratio:", value=0.8)

preprocessors = get_muss_preprocessors(length_ratio, replace_ratio, word_ratio, treedepth_ratio)
composed_preprocessor = ComposedPreprocessor(preprocessors)

text_a = c.text_input('Sentence to be simplified:',
                      value='This is an exquisite example sentence in which I am, exclusively, contemplating utter nonsense.',
                      max_chars=150)
if text_a != '':
    text = simplify(text_a)
    c.write(text)
c.markdown("***")

st.markdown("***")
st.markdown("#### Based on [MUSS: Multilingual Unsupervised Sentence Simplification by "
            "Mining Paraphrases](https://github.com/facebookresearch/muss)")
st.write("Paper: [Arxiv](https://arxiv.org/abs/2005.00352)")