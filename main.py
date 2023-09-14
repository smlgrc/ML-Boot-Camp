import time
import re
import os
import json
from io import TextIOWrapper
from typing import TextIO
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
OIE_CORPUS_FOLDER = r"C:\Users\Sam\Downloads\snapshot_oie_corpus.tar\snapshot_oie_corpus\oie_corpus"


def oie_parsing() -> tuple:
    # Define the path to your .oie file
    oie_file_path = os.path.join(OIE_CORPUS_FOLDER, "all.oie")

    # Initialize lists to store the extracted data
    data: list = []
    original_lines: list = []

    # Open the .oie file for reading
    with open(oie_file_path, 'r', encoding='utf-8') as oie_file:
        # Iterate through each line in the file
        for line in oie_file:
            # Split the line into fields using tab ('\t') as the delimiter
            original_lines.append(line)
            fields = line.strip().split('\t')

            # Check if the line has at least three fields (sentence, relation, and at least two arguments)
            if len(fields) >= 3:
                sentence, relation = fields[:2]
                args = fields[2:]
            else:
                print(f"Skipping line: {line}")

            data.append({
                "sentence": sentence,
                "relation": relation,
                "arguments": args
            })

    # Close the .oie file
    oie_file.close()

    return data, original_lines


def oie_to_text_file(data: list):
    file_lines: list = []

    # this appends the above extracted information to a list to be written
    for i in range(len(data)):
        file_lines.append(f"Extraction {i + 1}:\n")
        file_lines.append(f"Sentence: {data[i]['sentence']}\n")
        file_lines.append(f"Relation: {data[i]['relation']}\n")
        file_lines.append(f"Arguments: {', '.join(data[i]['arguments'])}\n")
        file_lines.append("\n")

    with open('OIE Parsing.txt', 'w', encoding="utf-8") as f:
        f.writelines(file_lines)


def oie_data() -> list:
    data, original_lines = oie_parsing()

    # For example, you can print the extracted information
    for i in range(5):
        print(f"Extraction {i + 1}:")
        print(f"Original line: {original_lines[i].strip()}")
        print(f"Sentence: {data[i]['sentence']}")
        print(f"Relation: {data[i]['relation']}")
        print(f"Arguments: {', '.join(data[i]['arguments'])}")
        print()

    # oie_to_text_file(data)
    # oie_to_json(data)

    return data


def oie_analysis(text: str = "", text_list: list = None):
    data: list = oie_data()

    average_sentence_length = round(sum(len(string) for string in text_list) / len(text_list))
    average_words_in_sentence = round(sum(util_func.num_of_words_in_str(sentence) for sentence in text_list) / len(text_list))
    average_arguments_per_sentence = round(sum(len(data_set['arguments']) for data_set in data) / len(data))
    print(f"Average sentence length = {average_sentence_length}")
    print(f"Average words in sentence = {average_words_in_sentence}")
    print(f"Average arguments per sentence = {average_arguments_per_sentence}")

    if text_list is not None:
        text: str = '\n'.join(text_list)

    # tokenize text by words
    words: list = word_tokenize(text)

    clean_words: list = util_func.get_filtered_words(words)

    processed_list: list = util_func.process_list(text_list)
    gensim_modeling(processed_list)

    # find the frequency of words
    fdist: nltk.probability.FreqDist = FreqDist(clean_words)

    # # Plot the 10 most common words
    # fdist.plot(20, title="derp", percents=True, show=True)
    # plt.show()

    sorted_list: list = sorted(fdist.items(), key=lambda item: item[1], reverse=True)

    util_func.plot_graph(sorted_list, 30)

    # Convert word list to a single string
    clean_words_string = " ".join(clean_words)

    # generating the wordcloud
    wordcloud = WordCloud(background_color="white").generate(clean_words_string)

    # plot the wordcloud
    plt.figure(figsize=(12, 12))
    plt.imshow(wordcloud)

    # # to remove the axis value
    plt.axis("off")
    plt.show()


def oie_to_json(data: list):
    unique_list: list = []
    for data_set in data:
        unique_list.append(data_set['sentence'])

    unique_list = list(set(unique_list))

    with open("OIE Sentences.json", 'w') as json_file:
        json.dump(unique_list, json_file)


def main():
    # retrieve text file from source, then read and decode the text
    # text: str = open("all.txt", "r", encoding='utf-8').read()
    text_list: list = util_func.load_json("OIE Sentences.json")

    oie_analysis(text_list=text_list)


if __name__ == '__main__':
    startTime = time.time()

    main()

    endTime = time.time()
    finalTime = (endTime - startTime)
    print("\nRunning Time:", "{:.2f}".format(finalTime) + " s")
