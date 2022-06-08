import re

SPECIAL_TOKEN_REGEX = r'<[a-zA-Z\-_\d\.]+>'

def extract_special_tokens(sentence):
    '''Remove any number of token at the beginning of the sentence'''
    match = re.match(fr'(^(?:{SPECIAL_TOKEN_REGEX} *)+) *(.*)$', sentence)
    if match is None:
        return '', sentence
    special_tokens, sentence = match.groups()
    return special_tokens.strip(), sentence

def remove_multiple_whitespaces(text):
    return re.sub(r'  +', ' ', text)

