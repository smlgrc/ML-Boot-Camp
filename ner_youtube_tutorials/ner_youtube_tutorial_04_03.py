#   NAMED ENTITY RECOGNITION SERIES   #
#             Lesson 04.03            #
#       Training a spaCy NER model    #
#               with                  #
#        Dr. W.J.B. Mattingly         #
import spacy
import json
import random


def load_data(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return (data)


def save_data(file, data):
    with open (file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def train_spacy(data, iterations):
    TRAIN_DATA = data
    nlp = spacy.blank("en")
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with (nlp.disable_pipes(*other_pipes)):
        optimizer = nlp.begin_training()
        for itn in range(iterations):
            print ("Starting iteration " + str(itn))
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in TRAIN_DATA:
                nlp.update(
                            [text],
                            [annotations],
                            drop=0.2,
                            sgd=optimizer,
                            losses=losses
                )
            print(losses)
    return nlp