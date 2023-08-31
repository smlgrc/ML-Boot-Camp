#   NAMED ENTITY RECOGNITION SERIES   #
#             Lesson 04.02            #
#        Leveraging spaCy's NER       #
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


def test_model(nlp_model, text):
    doc = nlp_model(text)
    results: list = []
    entities: list = []

    """
    Harry | (19, 24, 'PERSON') | "Oh, I will," said Harry, and they were surprised at the grin that was spreading over his face. "They don't know we're not allowed to use magic at home. I'm going to have a lot of fun with Dudley this summer...."
    Dudley | (189, 195, 'PERSON') | "Oh, I will," said Harry, and they were surprised at the grin that was spreading over his face. "They don't know we're not allowed to use magic at home. I'm going to have a lot of fun with Dudley this summer...."
    """
    # TRAIN_DATA = [(text, {"entities": [(start, end, label)]})]
    for ent in doc.ents:
        entity = (ent.start_char, ent.end_char, ent.label_)
        entities.append(entity)
        # print(f"{ent} | {entity} | {text}")
    if len(entities) > 0:  # if entity has been found
        results = [text, {"entities": entities}]
        return results


