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


def testing(matrix_A, matrix_B) -> np.ndarray:
    return matrix_A @ matrix_B


def extract_hp_characters():
    url = "https://en.wikipedia.org/wiki/List_of_Harry_Potter_characters"
    names = []
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "lxml")
    uls: bs4.element.ResultSet = soup.find_all("li")
    for ul in uls:
        line = str(ul.text)
        if "–" in line or "–" in line:
            name = line.split("–")[0].strip()
            if "The Deathly Hallows" in name:
                break
            else:
                if "/" in name:
                    two_names = name.split("/")
                    for item in two_names:
                        item = item.strip()
                        names.append(item)
                else:
                    names.append(name)
    print (names)
    with open ("ner_youtube_tutorials/data/hp_characters.json", "w", encoding="utf-8") as f:
        json.dump(names, f, indent=4)


def plot_graph(string_and_freq_list: list, n_terms: int):
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
    """ To use LDA, documents need to be represented as bag-of-words"""
    # create a dictionary from your preprocessed text data
    dictionary = corpora.Dictionary(processed_documents)

    # create a corpus (BoW representation) from your documents
    corpus = [dictionary.doc2bow(doc) for doc in processed_documents]

    """ To train LDA model, we need to specify the number of topics we want the model to discover"""
    lda_model = LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15)

    """ After training, we can inspect the discovered topics and the terms associated with each topic """
    for topic_num, topic_terms in lda_model.print_topics():
        print(f"Topic {topic_num}: {topic_terms}")

    """ Assign topics to individual documents in the corpus using the trained LDA model """
    documents_topics = [lda_model[doc] for doc in corpus]

    # print the topics assigned to a specific document (i.e. the first document)
    print(documents_topics[0])
    