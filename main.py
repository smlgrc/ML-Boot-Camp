import time
import re
import os
import json
from io import TextIOWrapper
from typing import TextIO

import OIE
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


def coreNLP_tutorial(text: str = None, corpus_path: str = None):
    from openie import StanfordOpenIE

    # Demo purposes
    if text is None:
        text = 'Barack Obama was born in Hawaii. Richard Manning wrote this sentence.'
    if corpus_path is None:
        corpus_path = 'corpus/pg6130.txt'

    # https://stanfordnlp.github.io/CoreNLP/openie.html#api
    # Default value of openie.affinity_probability_cap was 1/3.
    properties = {
        'openie.affinity_probability_cap': 2 / 3,
    }

    with StanfordOpenIE(properties=properties) as client:
        # text = 'Barack Obama was born in Hawaii. Richard Manning wrote this sentence.'
        print('Text: %s.' % text)
        for triple in client.annotate(text):
            print('|-', triple)

        # graph_image = 'graph.png'
        # client.generate_graphviz_graph(text, graph_image)
        # print('Graph generated: %s.' % graph_image)

        with open(corpus_path, encoding='utf8') as r:
            corpus = r.read().replace('\n', ' ').replace('\r', '')

        triples_corpus = client.annotate(corpus[0:5000])
        print('Corpus: %s [...].' % corpus[0:80])
        print('Found %s triples in the corpus.' % len(triples_corpus))
        for triple in triples_corpus[:5]:
            print('|-', triple)
        print('[...]')


def main():
    # corpus_path: str = "OIE Sentences.txt"
    # text: str = "My name is Samuel Garcia. I was born in California. I wrote this sentence."
    # coreNLP_tutorial(text=text, corpus_path=corpus_path)

    OIE.run_oie_analysis()

    # from allennlp.predictors.predictor import Predictor
    # import allennlp_models.tagging
    #
    # predictor = Predictor.from_path(
    #     "https://storage.googleapis.com/allennlp-public-models/openie-model.2020.03.26.tar.gz")
    # predictor.predict(
    #     sentence="In December, John decided to join the party.."
    # )


if __name__ == '__main__':
    start = time.time()

    main()

    end = time.time()
    finalTime = end - start
    print("\nProgram run time:", "{:.2f}".format(finalTime) + " s")
