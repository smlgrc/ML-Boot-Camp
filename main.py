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


def main():
    corpus_path: str = "OIE Sentences Original.txt"
    text: str = "My name is Samuel Garcia. I was born in California. I wrote this sentence."
    tutorials.coreNLP_tutorial(text=text, corpus_path=corpus_path)
    # OIE.run_oie_analysis()


if __name__ == '__main__':
    start = time.time()

    main()

    end = time.time()
    finalTime = end - start
    print("\nProgram run time:", "{:.2f}".format(finalTime) + " s")
