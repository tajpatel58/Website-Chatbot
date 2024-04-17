import json
import pandas as pd
from pathlib import Path
from nltk.stem import PorterStemmer
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from scripts.text_cleaning import clean_text, bow_text_to_tensor


class ChatbotDataset(Dataset):
    def __init__(self, data_path: Path, stemmer: PorterStemmer, stop_words: list):
        self.stemmer = stemmer
        self.data_path = data_path
        self.stop_words = stop_words
        self.set_label_encoder()

    def __getitem__(self, idx):
        return self.feature_tensors_list[idx], self.encoded_labels_list[idx]

    def __len__(self):
        return len(self.messages_list)

    def get_raw_data(self, data_path) -> dict:
        with open(data_path, "r") as f:
            message_json = json.loads(f.read())
        return message_json

    def load_data(self, data_path: Path) -> pd.DataFrame:
        raw_json_data = self.get_raw_data(data_path)

        questions_df_list = []
        messages_dict_list = raw_json_data["messages"]
        for question_grouping in messages_dict_list:
            question_df = pd.DataFrame({"question": question_grouping["question"]})
            question_df["question_type"] = question_grouping["question_type"]
            questions_df_list.append(question_df)
        question_label_df = pd.concat(questions_df_list, axis=0).reset_index(drop=True)
        return question_label_df

    def set_label_encoder(self):
        self.label_encoder = LabelEncoder()

    def get_label_encoder(self):
        return self.label_encoder

    def clean_data(self, question_label_df: pd.DataFrame) -> pd.DataFrame:
        # remove punctuation:
        cleaned_question_label_df = question_label_df
        cleaned_question_label_df[
            "question_no_punctuation"
        ] = cleaned_question_label_df["question"].str.replace(
            r"[^[a-zA-Z\s]", "", regex=True
        )

        # clean/tokenize text:
        cleaned_question_label_df["tokenized_question"] = cleaned_question_label_df[
            "question_no_punctuation"
        ].apply(clean_text, args=(self.stemmer, self.stop_words))

        # encode labels:
        cleaned_question_label_df[
            "question_type_encoded"
        ] = self.label_encoder.fit_transform(cleaned_question_label_df["question_type"])
        return cleaned_question_label_df

    def set_bag_of_words(self, cleaned_question_label_df: pd.DataFrame) -> set:
        self.bag_of_words = set(cleaned_question_label_df["tokenized_question"].sum())
        self.bag_size = len(self.bag_of_words)
        return self.bag_of_words, self.bag_size

    def get_bag_words(self):
        return self.bag_of_words, self.bag_size

    def set_features_and_labels(self, cleaned_question_label_df: pd.DataFrame) -> tuple:
        self.feature_tensors_list = [
            bow_text_to_tensor(tokenized_text, self.bag_of_words)
            for tokenized_text in cleaned_question_label_df["tokenized_question"]
        ]
        self.encoded_labels_list = cleaned_question_label_df[
            "question_type_encoded"
        ].tolist()
        return self.feature_tensors_list, self.encoded_labels_list

    def get_num_clases(self):
        num_classes = len(self.label_encoder.classes_)
        return num_classes

    def get_bag_size(self):
        return self.bag_size

    def load_and_process_data(self):
        raw_data_df = self.load_data(self.data_path)
        cleaned_questions_data = self.clean_data(raw_data_df)
        self.set_bag_of_words(cleaned_questions_data)
        self.set_features_and_labels(cleaned_questions_data)
