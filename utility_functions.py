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


def jsonprint(data: dict):
    print(json.dumps(data, indent=4))


def load_json(file_path: str) -> list:
    with open(file_path, 'r') as json_file:
        loaded_list = json.load(json_file)
    return loaded_list


def huggingface():
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    tokenizer = AutoTokenizer.from_pretrained("XLab/rst-information-extraction-11b")
    model = AutoModelForSeq2SeqLM.from_pretrained("XLab/rst-information-extraction-11b")

    inputs = tokenizer.encode(
        "TEXT: this is the best cast iron skillet you will ever buy. QUERY: Is this review \"positive\" or \"negative\"",
        return_tensors="pt")
    outputs = model.generate(inputs)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True))


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


def list_to_json(data: list, file_path: str):
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)


# def spacy_claucy():
#     import spacy
#     import claucy
#
#     nlp = spacy.load("en_core_web_sm")
#     claucy.add_to_pipe(nlp)
#
#     doc = nlp("AE died in Princeton in 1955.")
#
#     print(doc._.clauses)
#     # Output:
#     # &lt;SV, AE, died, None, None, None, [in Princeton, in 1955]&gt;
#
#     propositions = doc._.clauses[0].to_propositions(as_text=True)
#
#     breakpoint()
#
#     print(propositions)
#     # Output:
#     # [AE died in Princeton in 1955, AE died in 1955, AE died in Princeton