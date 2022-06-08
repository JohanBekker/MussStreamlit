# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC
from functools import wraps, lru_cache
import hashlib
from pathlib import Path
# import dill as pickle

from preprocessing.utils.text import extract_special_tokens
from preprocessing.utils.helpers import add_dicts, get_default_args
from preprocessing.utils.gpt2_utils import get_encoder

from preprocessing.utils.download_extract import download

PREPROCESSORS_REGISTRY = {}


def get_preprocessor_by_name(preprocessor_name):
    return PREPROCESSORS_REGISTRY[preprocessor_name]


def get_preprocessors(preprocessors_kwargs):
    preprocessors = []
    for preprocessor_name, kwargs in preprocessors_kwargs.items():
        preprocessors.append(get_preprocessor_by_name(preprocessor_name)(**kwargs))
    return preprocessors


def remove_special_tokens(sentence):
    return extract_special_tokens(sentence)[1]


def store_args(constructor):
    @wraps(constructor)
    def wrapped(self, *args, **kwargs):
        if not hasattr(self, 'args') or not hasattr(self, 'kwargs'):
            # TODO: Default args are not overwritten if provided as args
            self.args = args
            self.kwargs = add_dicts(get_default_args(constructor), kwargs)
        return constructor(self, *args, **kwargs)

    return wrapped


class AbstractPreprocessor(ABC):
    def __init_subclass__(cls, **kwargs):
        '''Register all children in registry'''
        super().__init_subclass__(**kwargs)
        PREPROCESSORS_REGISTRY[cls.__name__] = cls

    def __repr__(self):
        args = getattr(self, 'args', ())
        kwargs = getattr(self, 'kwargs', {})
        args_repr = [repr(arg) for arg in args]
        kwargs_repr = [f'{k}={repr(v)}' for k, v in sorted(kwargs.items(), key=lambda kv: kv[0])]
        args_kwargs_str = ', '.join(args_repr + kwargs_repr)
        return f'{self.__class__.__name__}({args_kwargs_str})'

    def get_hash_string(self):
        return self.__class__.__name__

    def get_hash(self):
        return hashlib.md5(self.get_hash_string().encode()).hexdigest()

    # @staticmethod
    # def get_nevergrad_variables():
    #     return {}

    @property
    def prefix(self):
        return self.__class__.__name__.replace('Preprocessor', '')

    def fit(self, complex_filepath, simple_filepath):
        pass

    def encode_sentence(self, sentence, encoder_sentence=None):
        raise NotImplementedError

    def decode_sentence(self, sentence, encoder_sentence=None):
        raise NotImplementedError


class ComposedPreprocessor(AbstractPreprocessor):
    @store_args
    def __init__(self, preprocessors, sort=False):
        if preprocessors is None:
            preprocessors = []
        if sort:
            # Make sure preprocessors are always in the same order
            preprocessors = sorted(preprocessors, key=lambda preprocessor: preprocessor.__class__.__name__)
        self.preprocessors = preprocessors

    def get_hash_string(self):
        preprocessors_hash_strings = [preprocessor.get_hash_string() for preprocessor in self.preprocessors]
        return f'ComposedPreprocessor(preprocessors={preprocessors_hash_strings})'

    def get_suffix(self):
        return '_'.join([p.prefix.lower() for p in self.preprocessors])

    def fit(self, complex_filepath, simple_filepath):
        for preprocessor in self.preprocessors:
            pass

    def encode_sentence(self, sentence, encoder_sentence=None):
        for preprocessor in self.preprocessors:
            sentence = preprocessor.encode_sentence(sentence, encoder_sentence)
        return sentence

    def decode_sentence(self, sentence, encoder_sentence=None):
        for preprocessor in self.preprocessors:
            sentence = preprocessor.decode_sentence(sentence, encoder_sentence)
        return sentence


class FeaturePreprocessor(AbstractPreprocessor):
    '''Prepend a computed feature at the beginning of the sentence'''

    @store_args
    def __init__(
            self,
            feature_name,
            # get_feature_value,
            get_target_feature_value,
            bucket_size=0.05,
            noise_std=0,
            prepend_to_target=False,
            use_short_name=False,
    ):
        # self.get_feature_value = get_feature_value
        self.get_target_feature_value = get_target_feature_value
        self.bucket_size = bucket_size
        self.noise_std = noise_std
        self.feature_name = feature_name.upper()
        self.use_short_name = use_short_name
        if use_short_name:
            # There might be collisions
            self.feature_name = self.feature_name.lower()[:4]
        self.prepend_to_target = prepend_to_target

    def get_hash_string(self):
        return f'{self.__class__.__name__}(feature_name={repr(self.feature_name)}, bucket_size={self.bucket_size}, noise_std={self.noise_std}, prepend_to_target={self.prepend_to_target}, use_short_name={self.use_short_name})'  # noqa: E501

    def bucketize(self, value):
        '''Round value to bucket_size to reduce the number of different values'''
        return round(round(value / self.bucket_size) * self.bucket_size, 10)

    # def add_noise(self, value):
    #     return value + np.random.normal(0, self.noise_std)

    def get_feature_token(self, feature_value):
        return f'<{self.feature_name}_{feature_value}>'

    def encode_sentence(self, sentence, encoder_sentence=None):
        if not self.prepend_to_target:
            desired_feature = self.bucketize(self.get_target_feature_value(remove_special_tokens(sentence)))
            sentence = f'{self.get_feature_token(desired_feature)} {sentence}'
        return sentence

    def decode_sentence(self, sentence, encoder_sentence=None):
        if self.prepend_to_target:
            _, sentence = extract_special_tokens(sentence)
        return sentence


class LevenshteinPreprocessor(FeaturePreprocessor):
    @store_args
    def __init__(self, target_ratio=0.8, bucket_size=0.05, noise_std=0, **kwargs):
        self.target_ratio = target_ratio
        super().__init__(
            self.prefix.upper(), self.get_target_feature_value, bucket_size, noise_std, **kwargs
        )

    # @staticmethod
    # def get_nevergrad_variables():
    #     return {'target_ratio': ng.p.Scalar(init=0.8, lower=0.2, upper=1)}

    # def get_feature_value(self, complex_sentence, simple_sentence):
    #     #return get_levenshtein_similarity(complex_sentence, simple_sentence)
    #     pass

    def get_target_feature_value(self, complex_sentence):
        return self.target_ratio


class ReplaceOnlyLevenshteinPreprocessor(LevenshteinPreprocessor):
    def get_feature_value(self, complex_sentence, simple_sentence):
        # return get_replace_only_levenshtein_similarity(complex_sentence, simple_sentence)
        pass


class RatioPreprocessor(FeaturePreprocessor):
    @store_args
    def __init__(self, target_ratio=0.8, bucket_size=0.05, noise_std=0, **kwargs):
        # self.feature_extractor = feature_extractor
        self.target_ratio = target_ratio
        super().__init__(
            self.prefix.upper(), self.get_target_feature_value, bucket_size, noise_std, **kwargs
        )

    # @staticmethod
    # def get_nevergrad_variables():
    #     return {'target_ratio': ng.p.Scalar(init=0.8, lower=0.2, upper=1.5)}

    # def get_feature_value(self, complex_sentence, simple_sentence):
    #     return min(
    #         failsafe_division(self.feature_extractor(simple_sentence), self.feature_extractor(complex_sentence)), 2
    #     )

    def get_target_feature_value(self, complex_sentence):
        return self.target_ratio


class LengthRatioPreprocessor(RatioPreprocessor):
    @store_args
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class WordRankRatioPreprocessor(RatioPreprocessor):
    @store_args
    def __init__(self, *args, language='en', **kwargs):
        super().__init__(*args, **kwargs)


class DependencyTreeDepthRatioPreprocessor(RatioPreprocessor):
    @store_args
    def __init__(self, *args, language='en', **kwargs):
        super().__init__(*args, **kwargs)

RESOURCES_DIR = Path('tokenizer/')

class GPT2BPEPreprocessor(AbstractPreprocessor):
    def __init__(self):
        self.bpe_dir = RESOURCES_DIR / 'bart_bpe'
        self.bpe_dir.mkdir(exist_ok=True, parents=True)
        self.encoder_json_path = self.bpe_dir / 'encoder.json'
        self.vocab_bpe_path = self.bpe_dir / 'vocab.bpe'
        self.dict_path = self.bpe_dir / 'dict.txt'
        download('https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json', self.encoder_json_path)
        download('https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe', self.vocab_bpe_path)
        download('https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt', self.dict_path)

    @property
    @lru_cache(maxsize=1)
    def bpe_encoder(self):
        """
        We need to use a property because GPT2BPEPreprocessor() is cannot pickled
        > pickle.dumps(GPT2BPEPreprocessor())
        ----> TypeError: can't pickle module objects
        """
        return get_encoder(self.encoder_json_path, self.vocab_bpe_path)

    def encode_sentence(self, sentence, *args, **kwargs):
        return ' '.join([str(idx) for idx in self.bpe_encoder.encode(sentence)])

    def decode_sentence(self, sentence, *args, **kwargs):
        return self.bpe_encoder.decode([int(idx) for idx in sentence.split(' ')])