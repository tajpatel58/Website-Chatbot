### Import Packages:
import torch

def clean_text(text, stemmer):
    lowered_text = text.lower()
    list_text = lowered_text.split(' ')
    clean_text = [stemmer.stem(word) for word in list_text]
    return clean_text 


def bag_of_words(tokenized_text, bag):
    feature_vec = torch.zeros(len(bag))
    for idx, word in enumerate(bag):
        if word in tokenized_text:
            feature_vec[idx] = 1
        else:
            continue
    return feature_vec