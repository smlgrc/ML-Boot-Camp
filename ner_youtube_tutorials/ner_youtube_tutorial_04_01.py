import json

from spacy import Language
from spacy.lang.en import English
from spacy.pipeline import EntityRuler

import ner_youtube_functions
import spacy


def load_data(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_data(file, data):
    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def generate_better_characters(file) -> list:
    data: list = load_data(file)
    print(len(data))
    new_characters: list = []
    for item in data:
        new_characters.append(item)
    for item in data:
        item: str = item.replace("The", "").replace("the", "").replace("and", "").replace("And", "")
        names: list = item.split(" ")
        for name in names:
            name = name.strip()
            new_characters.append(name)
        if "(" in item:
            names = item.split("(")
            for name in names:
                name = name.replace(")", "").strip()
                new_characters.append(name)
        if "," in item:
            names = item.split(",")
            for name in names:
                name = name.replace("and", "").strip()
                if " " in name:
                    new_names = name.split()
                    for x in new_names:
                        x = x.strip()
                        new_characters.append(x)
                new_characters.append(name)
    print(len(new_characters))
    final_characters: list = []
    titles = ["Dr.", "Professor", "Mr.", "Mrs.", "Ms.", "Miss", "Aunt", "Uncle", "Mr. and Mrs."]
    # creates variations of a name with a title appended to it
    for character in new_characters:
        if "" != character:
            final_characters.append(character)
            for title in titles:
                titled_char = f"{title} {character}"
                final_characters.append(titled_char)

    print(len(final_characters))
    final_characters = list(set(final_characters))  # gets rid of duplicates
    print(len(final_characters))
    final_characters.sort()
    return final_characters


def create_training_data(file: str, ent_type: str) -> list:
    data: list = generate_better_characters(file)
    # spaCy expects to see a dictionary that consists of two things, a label and a pattern
    patterns: list = []
    for item in data:
        pattern = {
            "label": ent_type,
            "pattern": item
        }
        patterns.append(pattern)
    return patterns


def get_ent_ruler(nlp, patterns):
    ruler = EntityRuler(nlp)  # Used for entity recognition, allowing for defining patterns that match exact entities
    ruler.add_patterns(patterns)  # Patterns define what the ruler should look for in the text to identify entities
    return ruler


def generate_rules_EXPERIMENTAL(patterns: list):
    nlp = English()  # A spaCy language processing pipeline initialized with an English language model
    Language.factory("ent_ruler", func=get_ent_ruler(nlp, patterns))
    nlp.add_pipe("entity_ruler")  # This step integrates the custom entity recognition rules defined in ruler into pipeline
    nlp.to_disk("hp_ner")  # Save spaCy pipeline, including custom ruler to disk


def generate_rules_OLD(patterns: list):
    nlp = English()  # A spaCy language processing pipeline initialized with an English language model
    ruler = EntityRuler(nlp)  # Used for entity recognition, allowing for defining patterns that match exact entities
    ruler.add_patterns(patterns)  # Patterns define what the ruler should look for in the text to identify entities
    nlp.add_pipe(ruler)  # This step integrates the custom entity recognition rules defined in ruler into pipeline
    nlp.to_disk("hp_ner")  # Save spaCy pipeline, including custom ruler to disk


def generate_rules(patterns: list):
    nlp: spacy.lang.en.English = English()  # A spaCy language processing pipeline initialized with an English language model
    # ruler = EntityRuler(nlp)  # Used for entity recognition, allowing for defining patterns that match exact entities
    # ruler.add_patterns(patterns)  # Patterns define what the ruler should look for in the text to identify entities
    ruler: spacy.pipeline.entityruler.EntityRuler = nlp.add_pipe("entity_ruler")
    ruler.add_patterns(patterns)

    # nlp.add_pipe(ruler)  # This step integrates the custom entity recognition rules defined in ruler into pipeline
    nlp.to_disk("hp_ner_folder/hp_ner")  # Save spaCy pipeline, including custom ruler to disk


def test_model(nlp_model, text):
    doc = nlp_model(text)
    results = []
    for ent in doc.ents:
        results.append(ent.text)
    return results


