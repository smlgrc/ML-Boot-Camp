import pandas as pd
import sklearn
from sklearn.datasets import fetch_openml
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np

import time
import json
from matplotlib import pyplot as plt

import ner_youtube_functions
import tutorials
import utility_functions as util_func
import spacy


def ner_youtube_tutorial():
    # https://youtu.be/wpyCzodvO3A?si=ACE1qNqb3_b1XETP&t=423
    patterns = ner_youtube_functions.create_training_data("data/hp_characters.json", "PERSON")
    ner_youtube_functions.generate_rules(patterns)
    # print (patterns)

    nlp = spacy.load("hp_ner")
    ie_data = {}
    with open("data/hp.txt", "r") as f:
        text = f.read()

        chapters = text.split("CHAPTER")[1:]
        for chapter in chapters:
            chapter_num, chapter_title = chapter.split("\n\n")[0:2]
            chapter_num = chapter_num.strip()
            segments = chapter.split("\n\n")[2:]
            hits = []
            for segment in segments:
                segment = segment.strip()
                segment = segment.replace("\n", " ")
                results = ner_youtube_functions.test_model(nlp, segment)
                for result in results:
                    hits.append(result)
            ie_data[chapter_num] = hits

    ner_youtube_functions.save_data("data/hp_data.json", ie_data)


def main():
    # nn_from_scratch()
    # tutorials.patrick_loeber_tutorial()
    # tutorials.samson_zhang_nn()
    ner_youtube_tutorial()


if __name__ == '__main__':
    startTime = time.time()
    main()
    endTime = time.time()
    finalTime = (endTime - startTime)

    print("\nRunning Time:", "{:.2f}".format(finalTime) + " s")
