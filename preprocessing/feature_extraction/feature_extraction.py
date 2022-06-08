from functools import lru_cache

import numpy as np
from pathlib import Path

from preprocessing.utils.helpers import yield_lines
from preprocessing.utils.download_extract import download_and_extract

FASTTEXT_EMBEDDINGS_DIR = Path('fasttext-vectors/')


def get_fasttext_embeddings_path(language='en'):
    fasttext_embeddings_path = FASTTEXT_EMBEDDINGS_DIR / f'cc.{language}.300.vec'
    if not fasttext_embeddings_path.exists():
        url = f'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.{language}.300.vec.gz'
        fasttext_embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        download_and_extract(url, fasttext_embeddings_path)
        # shutil.move(download_and_extract(url)[0], fasttext_embeddings_path)
    return fasttext_embeddings_path


@lru_cache(maxsize=10)
def get_word2rank(vocab_size=10 ** 5, language='en'):
    word2rank = {}
    line_generator = yield_lines(get_fasttext_embeddings_path(language))
    next(line_generator)  # Skip the first line (header)
    for i, line in enumerate(line_generator):
        if (i + 1) > vocab_size:
            break
        word = line.split(' ')[0]
        word2rank[word] = i
    return word2rank


def get_rank(word, language='en'):
    return get_word2rank(language=language).get(word, len(get_word2rank(language=language)))


def get_log_rank(word, language='en'):
    return np.log(1 + get_rank(word, language=language))


def get_log_ranks(text, language='en'):
    return [
        get_log_rank(word, language=language)
        for word in get_content_words(text, language=language)
        if word in get_word2rank(language=language)
    ]


# Single sentence feature extractors with signature function(sentence) -> float
def get_lexical_complexity_score(sentence, language='en'):
    log_ranks = get_log_ranks(sentence, language=language)
    if len(log_ranks) == 0:
        log_ranks = [np.log(1 + len(get_word2rank()))]  # TODO: This is completely arbitrary
    return np.quantile(log_ranks, 0.75)
