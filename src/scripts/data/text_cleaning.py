import torch
import string
from nltk.stem import PorterStemmer


def clean_text(text: str, stemmer: PorterStemmer, stop_words: list) -> list:
    lowered_text = text.lower()
    text_no_punctuation = lowered_text.translate(str.maketrans('', '', string.punctuation))
    list_text = text_no_punctuation.split()
    clean_text = [stemmer.stem(word) for word in list_text if word not in stop_words]
    return clean_text


def bow_text_to_tensor(tokenized_text: list, bag: list) -> torch.Tensor:
    feature_vec = torch.zeros(len(bag))
    for idx, word in enumerate(bag):
        if word in tokenized_text:
            feature_vec[idx] = 1
        else:
            continue
    return feature_vec
