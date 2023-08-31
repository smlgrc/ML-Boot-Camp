import json
import spacy
from spacy import Language
from spacy.lang.en import English
from spacy.pipeline import EntityRuler
import ner_youtube_tutorials.ner_youtube_tutorial_04_01 as yt_0401
import ner_youtube_tutorials.ner_youtube_tutorial_04_02 as yt_0402


def ner_youtube_tutorial_0402():
    # Generate training data
    # https://youtu.be/YBRF7tq1V-Q?si=17Y_nSHXo3Dhnmm6

    # TRAIN_DATA = [(text, {"entities": [(start, end, label)]})]
    nlp = spacy.load("hp_ner")
    TRAIN_DATA: list = []
    with open("ner_youtube_data/hp.txt", "r") as f:
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
                results = yt_0402.test_model(nlp, segment)
                if results != None:
                    TRAIN_DATA.append(results)
    print(len(TRAIN_DATA))
    yt_0402.save_data("ner_youtube_data/hp_training_data.json", TRAIN_DATA)


def ner_youtube_tutorial_0401():
    #
    # https://youtu.be/wpyCzodvO3A?si=ACE1qNqb3_b1XETP&t=423
    patterns: list = yt_0401.create_training_data("ner_youtube_data/hp_characters.json", "PERSON")
    yt_0401.generate_rules(patterns)
    # print (patterns)

    nlp = spacy.load("hp_ner")
    ie_data = {}
    with open("ner_youtube_data/hp.txt", "r") as f:
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
                results = yt_0401.test_model(nlp, segment)
                for result in results:
                    hits.append(result)
            ie_data[chapter_num] = hits

    yt_0401.save_data("ner_youtube_data/hp_data.json", ie_data)


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
