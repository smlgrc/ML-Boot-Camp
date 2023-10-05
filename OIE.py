import os

import bs4
import numpy as np
import pandas as pd
import altair as alt
from nltk.corpus import stopwords
from sympy import Matrix
import requests
from bs4 import BeautifulSoup
import json
from matplotlib import pyplot as plt
import nltk
from nltk import word_tokenize
from nltk.probability import FreqDist
from gensim import corpora
from gensim.models import LdaModel
# nltk.download("stopwords")
from wordcloud import WordCloud
import utility_functions as util_func


OIE_CORPUS_FOLDER = r"C:\Users\Sam\Downloads\snapshot_oie_corpus.tar\snapshot_oie_corpus\oie_corpus"


def generate_wordcloud(words: str):
    # generating the wordcloud
    wordcloud = WordCloud(background_color="white").generate(words)

    # plot the wordcloud
    plt.figure(figsize=(12, 12))
    plt.imshow(wordcloud)

    # # to remove the axis value
    plt.axis("off")
    plt.show()


def generate_plot(string_and_freq_list: list, n_terms: int):
    x = [string_and_freq_list[i][0] for i in range(n_terms)]
    y = [string_and_freq_list[i][1] for i in range(n_terms)]

    # create a figure and axis object;
    fig, ax1 = plt.subplots(figsize=(8.5, 11))  # figsize=(50, 40))
    # fig, ax1 = plt.subplots()  # figsize=(50, 40))

    # plot the line graph:
    ax1.plot(x, y)

    # set the x axis tick labels to the strings from the tuples:
    ax1.set_xticklabels(x)

    # rotate the x axis tick labels:
    plt.xticks(rotation=45, ha='right')  # 0 means flat

    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0.5)

    plt.show()

    # Close the figure
    plt.close()


def get_github_stopwords() -> list:
    gist_file = open("github_stopwords.txt", "r")
    try:
        content = gist_file.read()
        stopwords = content.split(",")
        stopwords = [i.replace('"', "").strip() for i in stopwords]
    finally:
        gist_file.close()
    return stopwords


def get_filtered_words(words: list) -> list:
    nltk_stopwords_list: list = stopwords.words("english")
    github_stopwords_list: list = get_github_stopwords()
    stopwords_list: list = list(set(nltk_stopwords_list + github_stopwords_list))

    # lower case all words in list
    words = [word.lower() for word in words]

    # create a list to store alphabetic words only
    clean_words: list = []

    # iterate through the words list to remove punctuations
    for word in words:
        if word.isalpha() and word not in stopwords_list:
            clean_words.append(word)

    return clean_words


def load_json(file_path: str) -> list:
    with open(file_path, 'r') as json_file:
        loaded_list = json.load(json_file)
    return loaded_list


def generate_words_list(sentence: str, filter_words: bool = None) -> list:
    # Split the text into words based on spaces and punctuation
    # Filter out non-alphabetical and non-numerical characters
    word_list = [word.strip('-`.,!?"\'()[]{}') for word in sentence.split()]
    words_only_list = [word for word in word_list if word]  # remove empty strings

    if filter_words:
        words_only_list = get_filtered_words(words_only_list)

    return words_only_list


def num_of_words_in_str(text: str) -> int:
    filtered_word_list = generate_words_list(text)
    # count and return number of words in string if they're not an empty string
    return len(filtered_word_list)


def process_list(text_list: list) -> list:
    processed_list: list = []
    for sentence in text_list:
        processed_list.append(generate_words_list(sentence, filter_words=True))
    return processed_list


def gensim_modeling(processed_documents: list):
    """
    Average sentence length = 141
    Average words in sentence = 23
    Average arguments per sentence = 3
    Topic 0: 0.004*"company" + 0.003*"city" + 0.002*"years" + 0.002*"office" + 0.002*"district" + 0.002*"japan" + 0.002*"church" + 0.002*"album" + 0.002*"state" + 0.002*"south"
    Topic 1: 0.005*"company" + 0.005*"president" + 0.004*"share" + 0.003*"market" + 0.002*"chief" + 0.002*"officer" + 0.002*"american" + 0.002*"month" + 0.002*"number" + 0.002*"government"
    Topic 2: 0.006*"stock" + 0.005*"market" + 0.004*"billion" + 0.004*"board" + 0.004*"year" + 0.004*"trading" + 0.003*"program" + 0.003*"time" + 0.003*"big" + 0.002*"funds"
    Topic 3: 0.007*"years" + 0.004*"age" + 0.004*"living" + 0.004*"government" + 0.003*"days" + 0.003*"public" + 0.002*"company" + 0.002*"moved" + 0.002*"relationships" + 0.002*"work"
    Topic 4: 0.006*"year" + 0.003*"years" + 0.003*"time" + 0.002*"mortgage" + 0.002*"point" + 0.002*"rate" + 0.002*"yield" + 0.002*"season" + 0.002*"high" + 0.002*"school"
    [(0, 0.04011481), (1, 0.8380117), (2, 0.040187854), (3, 0.040838517), (4, 0.04084712)]
    """

    """ To use LDA, documents need to be represented as bag-of-words """
    # create a dictionary from your preprocessed text data
    dictionary = corpora.Dictionary(processed_documents)

    # create a corpus (BoW representation) from your documents
    corpus = [dictionary.doc2bow(doc) for doc in processed_documents]

    """ To train LDA model, we need to specify the number of topics we want the model to discover """
    lda_model = LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15)

    """ After training, we can inspect the discovered topics and the terms associated with each topic """
    for topic_num, topic_terms in lda_model.print_topics():
        print(f"Topic {topic_num}: {topic_terms}")

    """ Assign topics to individual documents in the corpus using the trained LDA model """
    documents_topics = [lda_model[doc] for doc in corpus]

    # print the topics assigned to a specific document (i.e. the first document)
    print(documents_topics[0])


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
    for i in range(10):
        print(f"Extraction {i + 1}:")
        print(f"Original line: {original_lines[i].strip()}")
        print(f"Sentence: {data[i]['sentence']}")
        print(f"Relation: {data[i]['relation']}")
        print(f"Arguments ({len(data[i]['arguments'])}): {' [*] '.join(data[i]['arguments'])}")
        print()

    # oie_to_text_file(data)
    # oie_to_json(data)

    return data


def oie_analysis(text: str = "", text_list: list = None):
    data: list = oie_data()

    average_sentence_length = round(sum(len(string) for string in text_list) / len(text_list))
    average_words_in_sentence = round(
        sum(num_of_words_in_str(sentence) for sentence in text_list) / len(text_list))
    average_arguments_per_sentence = round(sum(len(data_set['arguments']) for data_set in data) / len(data))
    print(f"Average sentence length = {average_sentence_length}")
    print(f"Average words in sentence = {average_words_in_sentence}")
    print(f"Average arguments per sentence = {average_arguments_per_sentence}")

    if text_list is not None:
        text: str = '\n'.join(text_list)

    # tokenize text by words
    words: list = word_tokenize(text)

    clean_words: list = get_filtered_words(words)

    processed_list: list = process_list(text_list)
    gensim_modeling(processed_list)

    # find the frequency of words
    fdist: nltk.probability.FreqDist = FreqDist(clean_words)

    sorted_list: list = sorted(fdist.items(), key=lambda item: item[1], reverse=True)

    # # Plot the 30 most common words
    # fdist.plot(30, title="derp", percents=True, show=True)
    # plt.show()
    generate_plot(string_and_freq_list=sorted_list, n_terms=30)

    # Convert word list to a single string and generate wordcloud
    clean_words_string: str = " ".join(clean_words)
    generate_wordcloud(words=clean_words_string)


def oie_to_json(data: list):
    unique_list: list = []
    for data_set in data:
        unique_list.append(data_set['sentence'])

    unique_list = list(set(unique_list))

    with open("OIE Sentences.json", 'w') as json_file:
        json.dump(unique_list, json_file)


def oie_to_text(data: list):
    with open('OIE Sentences.txt', 'w') as file:
        for sentence in data:
            file.write(sentence + '\n')


def run_oie_analysis():
    # retrieve text file from source, then read and decode the text
    # text: str = open("all.txt", "r", encoding='utf-8').read()
    text_list: list = load_json("OIE Sentences Original.json")
    oie_analysis(text_list=text_list)
