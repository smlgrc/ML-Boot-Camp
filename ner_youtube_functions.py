import json
import spacy


def load_data(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_data(file, data):
    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def generate_better_characters(file):
    data = load_data(file)
    print(len(data))
    new_characters = []
    for item in data:
        new_characters.append(item)
    for item in data:
        item = item.replace("The", "").replace("the", "").replace("and", "").replace("And", "")
        names = item.split(" ")
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
    final_characters = []
    titles = ["Dr.", "Professor", "Mr.", "Mrs.", "Ms.", "Miss", "Aunt", "Uncle", "Mr. and Mrs."]
    for character in new_characters:
        if "" != character:
            final_characters.append(character)
            for title in titles:
                titled_char = f"{title} {character}"
                final_characters.append(titled_char)

    print(len(final_characters))
    final_characters = list(set(final_characters))
    print(len(final_characters))
    final_characters.sort()
    return final_characters


def create_training_data(file, type):
    data = generate_better_characters(file)
    patterns = []
    for item in data:
        pattern = {
            "label": type,
            "pattern": item
        }
        patterns.append(pattern)
    return (patterns)


def generate_rules(patterns):
    nlp = English()
    ruler = EntityRuler(nlp)
    ruler.add_patterns(patterns)
    nlp.add_pipe(ruler)
    nlp.to_disk("hp_ner")


def test_model(model, text):
    doc = nlp(text)
    results = []
    for ent in doc.ents:
        results.append(ent.text)
    return (results)


def basic_spacy_use():
    #   NAMED ENTITY RECOGNITION SERIES   #
    #             Lesson 03               #
    #        Machine Learning NER         #
    #               with                  #
    #        Dr. W.J.B. Mattingly         #

    test = ("Mr. Dursley was the director of a firm called Grunnings, which made drills. He was a big, beefy man with "
            "hardly any neck, although he did have a very large mustache. Mrs. Dursley was thin and blonde and had "
            "nearly twice the usual amount of neck, which came in very useful as she spent so much of her time "
            "craning over garden fences, spying on the neighbors. The Dursleys had a small son called Dudley and in "
            "their opinion there was no finer boy anywhere.")

    nlp = spacy.load("en_core_web_lg")  # loading spacy machine learning model
    doc = nlp(test)
    for ent in doc.ents:  # for each entity in the ents pipeline
        print(ent.text, ent.label_)


def rule_based_NER():
    """
    not very effective
    """
    with open("ner_youtube_data/hp.txt", "r", encoding="utf-8") as f:
        text = f.read().split("\n\n")
        print(text)

    character_names = []
    with open("ner_youtube_data/hp_characters.json", "r", encoding="utf-8") as f:
        characters = json.load(f)
        for character in characters:
            names = character.split()
            for name in names:
                if "and" != name and "the" != name and "The" != name:
                    name = name.replace(",", "").strip()
                    character_names.append(name)

    for segment in text:
        # print (segment)
        segment = segment.strip()
        segment = segment.replace("\n", " ")
        print(segment)

        punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
        for ele in segment:
            if ele in punc:
                segment = segment.replace(ele, "")
        # print (segment)
        words = segment.split()
        # print (words)
        i = 0
        for word in words:
            if word in character_names:
                if words[i - 1][0].isupper():
                    print(f"Found Character(s): {words[i - 1]} {word}")
                else:
                    print(f"Found Character(s): {word}")

            i = i + 1
