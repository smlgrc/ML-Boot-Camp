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

def spacy_claucy():
    import spacy
    import claucy

    nlp = spacy.load("en_core_web_sm")
    claucy.add_to_pipe(nlp)

    doc = nlp("AE died in Princeton in 1955.")

    print(doc._.clauses)
    # Output:
    # &lt;SV, AE, died, None, None, None, [in Princeton, in 1955]&gt;

    propositions = doc._.clauses[0].to_propositions(as_text=True)

    breakpoint()

    print(propositions)
    # Output:
    # [AE died in Princeton in 1955, AE died in 1955, AE died in Princeton


def ollie_tutorial():
    sentence = "Barack Obama was born in Hawaii."
    try:
        # Run Ollie as a subprocess
        result = subprocess.run(
            ["java", "-Xmx1g", "-jar", "ollie/ollie-app-latest.jar", "corpus/sample.txt"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        print("here1")

        # Extracted information is in the standard output
        output = result.stdout

        print("here2")

        # Split the output into lines and extract relevant information
        lines = output.strip().split("\n")

        breakpoint()

        subject = lines[0]
        predicate = lines[1]
        arguments = lines[2:]

        print("here3")

        print(subject, predicate, arguments)
        return subject, predicate, arguments
    except Exception as e:
        print("Error:", str(e))
        return None, None, []


def huggingface():
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    tokenizer = AutoTokenizer.from_pretrained("XLab/rst-information-extraction-11b")
    model = AutoModelForSeq2SeqLM.from_pretrained("XLab/rst-information-extraction-11b")

    inputs = tokenizer.encode(
        "TEXT: this is the best cast iron skillet you will ever buy. QUERY: Is this review \"positive\" or \"negative\"",
        return_tensors="pt")
    outputs = model.generate(inputs)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True))


def main():
    # corpus_path: str = "OIE Sentences Original.txt"
    # text: str = "My name is Samuel Garcia. I was born in California. I wrote this sentence."
    # tutorials.coreNLP_tutorial(text=text, corpus_path=corpus_path)

    # OIE.run_oie_analysis()

    # spacy_claucy()
    # huggingface()
    ollie_tutorial()




if __name__ == '__main__':
    start = time.time()

    main()

    end = time.time()
    finalTime = end - start
    print("\nProgram run time:", "{:.2f}".format(finalTime) + " s")
