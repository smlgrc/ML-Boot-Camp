#   NAMED ENTITY RECOGNITION SERIES   #
#             Lesson 04.03            #
#       Training a spaCy NER model    #
#               with                  #
#        Dr. W.J.B. Mattingly         #
import spacy
import json
import random
import re
from spacy.training.example import Example


def load_data(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_data(file, data):
    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def clean_text(text):
    cleaned = re.sub(r"[\(\[].*?[\)\]]", "", text)
    return cleaned


def train_spacy(TRAIN_DATA: list, iterations: int):
    nlp = spacy.blank("en")  # fresh blank english model
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe("ner", last=True)
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])  # [0, 11, 'PERSON'] grabs 'PERSON'
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with (nlp.disable_pipes(*other_pipes)):  # disable_pipes makes it so that other pipes aren't affected
        optimizer = nlp.begin_training()
        for itn in range(iterations):
            print("Starting iteration " + str(itn))
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in TRAIN_DATA:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                nlp.update([example], drop=0.2, sgd=optimizer, losses=losses)
            print(losses)
    return nlp