import time
import re
import os
import json
from io import TextIOWrapper
from typing import TextIO

import OIE
import tutorials
import utility_functions as util_func

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

# import sklearn
# from sklearn.datasets import fetch_openml
# from keras.utils import to_categorical
# from sklearn.model_selection import train_test_split
# import ner_youtube_functions
# import tutorials

# import random
# import ner_youtube_tutorials.ner_youtube_tutorial_04_01 as yt_0401
# import ner_youtube_tutorials.ner_youtube_tutorial_04_02 as yt_0402
# import ner_youtube_tutorials.ner_youtube_tutorial_04_03 as yt_0403
# import spacy

# OIE analysis libraries
import nltk
from nltk import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
import urllib.request
from matplotlib import pyplot as plt
from wordcloud import WordCloud
# # is a pre-trained ML model that NLTK uses for tokenization tasks, for sentence and word tokenization
# nltk.download('punkt')

# NYT_PATH = 'ner_youtube_tutorials'
# NYT_DATA_PATH = os.path.join(NYT_PATH, "data")
# NYT_HP_FOLDER_PATH = os.path.join(NYT_PATH, "hp_folders")
# OIE_CORPUS_FOLDER = r"C:\Users\Sam\Downloads\snapshot_oie_corpus.tar\snapshot_oie_corpus\oie_corpus"


import subprocess


def allenNLP(sentence: str, json_list: list):
    """
    (Pdb) line
    'The effect is that lawsuits that might have been barred because they were filed too late could proceed because of the one - year extension .\tbarred\tmight have been barred\tlawsuits\tbecause they were filed too late could proceed because of the one - year extension\n'
    (Pdb) line.strip().split('\t')
    ['The effect is that lawsuits that might have been barred because they were filed too late could proceed because of the one - year extension .', 'barred', 'might have been barred', 'lawsuits', 'because they were filed too late could proceed because of the one - year extension']
    """
    from allennlp.predictors.predictor import Predictor

    # Load the OpenIE predictor
    predictor = Predictor.from_path(
        "https://storage.googleapis.com/allennlp-public-models/openie-model.2020.03.26.tar.gz")
    # # Load the OpenIE predictor with the BERT-based SRL model
    # predictor = Predictor.from_path(
    #     "https://storage.googleapis.com/allennlp-public-models/bert-base-srl-2020.11.19.tar.gz")

    # Extract relations and arguments
    output = predictor.predict(sentence=sentence)

    # Extract subjects, relations, and objects from the descriptions
    for verb_info in output['verbs']:
        # subject = ""
        verb = ""
        arguments = []

        description = verb_info['description']
        verb = verb_info['verb']

        # Use regular expressions to find all ARGs with numbers attached and their values
        arg_pattern = r'\[ARG(\d+): (.*?)\]'
        arg_matches = re.findall(arg_pattern, description)

        # Create a dictionary to store the extracted arguments and values
        arg_dict = {arg_num: arg_value for arg_num, arg_value in arg_matches}

        # Sort the dictionary by keys
        arg_dict = dict(sorted(arg_dict.items()))

        # Print the extracted argument and value pairs
        if verb:
            arguments.append(verb)  # adding verb to arguments because it's added to text.oie
        for arg_num, arg_value in arg_dict.items():
            if arg_num.isdigit():
                arguments.append(arg_value)

        # print(f"\nDescription: {description}")
        # print(f"arg_dict = {arg_dict}")
        # print(f"Subject: {subject}")
        print(f"\nOIE Extraction: {len(json_list) + 1}")
        print(f"Sentence: {sentence}")
        print(f"Predicate: {verb}")
        print(f"Arguments: {arguments}")
        json_list.append({
            "sentence": sentence,
            "predicate": verb,
            "arguments": arguments
        })


def main():
    # corpus_path: str = "OIE Sentences Original.txt"
    # text: str = "My name is Samuel Garcia. I was born in California. I wrote this sentence."
    # tutorials.coreNLP_tutorial(text=text, corpus_path=corpus_path)

    # OIE.run_oie_analysis()  # Outdated

    # spacy_claucy()
    # huggingface()
    # ollie_tutorial()

    # sentences = [
    #     "The effect is that lawsuits that might have been barred because they were filed too late could proceed because of the one - year extension .",
    #     "Mrs. Marcos has n't admitted that she filed any documents such as those sought by the government .",
    #     "Metromedia , headed by John W. Kluge , has interests in telecommunications , robotic painting , computer software , restaurants and entertainment .",
    #     "In his speech , the senator spoke of cutting the deficit .",
    #     "Barack Obama was born in Hawaii .",
    #     "Repeat customers also can purchase luxury items at reduced prices .",
    # ]
    sentences: list = util_func.load_json("OIE Sentences.json")
    json_list: list = []
    for sentence in sentences:
        allenNLP(sentence, json_list)
    util_func.list_to_json(json_list, "AllenNLP Extract.json")


if __name__ == '__main__':
    start = time.time()

    main()

    end = time.time()
    finalTime = end - start
    print("\nProgram run time:", "{:.2f}".format(finalTime) + " s")
