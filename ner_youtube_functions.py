import json
import spacy
from spacy import Language
from spacy.lang.en import English
from spacy.pipeline import EntityRuler
import ner_youtube_tutorials.ner_youtube_tutorial_04_01 as yt_0401
import ner_youtube_tutorials.ner_youtube_tutorial_04_02 as yt_0402
import ner_youtube_tutorials.ner_youtube_tutorial_04_03 as yt_0403


def ner_youtube_tutorial_0403():
    #  use the training set created in 04.01, to train an NER (named entity recognition)
    #  machine learning (ML) model via spaCy's NER pipe
    # https://youtu.be/7Z1imsp6g10?si=D4qDVeyUhmElUqTv&t=641

    """ Keep commented out because it will retrain the data and save it"""
    # TRAIN_DATA: list = yt_0403.load_data("ner_youtube_tutorials/data/hp_training_data.json")
    # nlp = yt_0403.train_spacy(TRAIN_DATA, 30)
    # nlp.to_disk("ner_youtube_tutorials/hp_folders/hp_ner_model")

    test = "Harry James[59] Potter (b. 31 July 1980[1]) was an English half-blood[2] wizard, and one of the most famous wizards of modern times. He was the only child and son of James and Lily Potter (née Evans), both members of the original Order of the Phoenix. Harry's birth was overshadowed by a prophecy, naming either himself or Neville Longbottom as the one with the power to vanquish Lord Voldemort. After half of the prophecy was reported to Voldemort, courtesy of Severus Snape, Harry was chosen as the target due to his many similarities with the Dark Lord. In turn, this caused the Potter family to go into hiding. Voldemort made his first vain attempt to circumvent the prophecy when Harry was a year and three months old. During this attempt, he murdered Harry's parents as they tried to protect him, but this unsuccessful attempt to kill Harry led to Voldemort's first downfall. This downfall marked the end of the First Wizarding War, and to Harry henceforth being known as the 'Boy Who Lived',[5] as he was the only known survivor of the Killing Curse. One consequence of Lily's loving sacrifice was that her orphaned son had to be raised by her only remaining blood relative, his Muggle aunt, Petunia Dursley. While in her care he would be protected from Lord Voldemort, due to the Bond of Blood charm Albus Dumbledore placed upon him.[60] This powerful charm would protect him until he became of age, or no longer called his aunt's house home. Due to Petunia's resentment of her sister and her magic gifts, Harry grew up abused and neglected. On his eleventh birthday, Harry learned that he was a wizard, from Rubeus Hagrid.[61] He began attending Hogwarts School of Witchcraft and Wizardry in 1991. The Sorting Hat was initially going to Sort Harry into Slytherin House, but Harry pleaded 'not Slytherin' and the Hat heeded this plea, instead sorting the young wizard into Gryffindor House.[62] At school, Harry became best friends with Ron Weasley and Hermione Granger. He later became the youngest Quidditch Seeker in over a century and eventually the captain of the Gryffindor House Quidditch Team in his sixth year, winning two Quidditch Cups.[63] He became even better known in his early years for protecting the Philosopher's Stone from Voldemort, saving Ron's sister Ginny Weasley, solving the mystery of the Chamber of Secrets, slaying Salazar Slytherin's basilisk, and learning how to conjure a corporeal stag Patronus at the age of thirteen. In his fourth year, Harry won the Triwizard Tournament, although the competition ended with the tragic death of Cedric Diggory and the return of Lord Voldemort. During the next school year, Harry reluctantly taught and led Dumbledore's Army. He also fought in the Battle of the Department of Mysteries, during which he lost his godfather, Sirius Black."
    test = yt_0403.clean_text(test)

    nlp = spacy.load("ner_youtube_tutorials/hp_folders/hp_ner_model")

    doc = nlp(test)
    for ent in doc.ents:
        print(ent.text, ent.label_)


def ner_youtube_tutorial_0402():
    # Generate training data
    # https://youtu.be/YBRF7tq1V-Q?si=17Y_nSHXo3Dhnmm6

    # TRAIN_DATA = [(text, {"entities": [(start, end, label)]})]
    nlp = spacy.load("ner_youtube_tutorials/hp_folders/hp_ner")
    TRAIN_DATA: list = []
    with open("ner_youtube_tutorials/data/hp.txt", "r") as f:
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
    yt_0402.save_data("ner_youtube_tutorials/data/hp_training_data.json", TRAIN_DATA)


def ner_youtube_tutorial_0401():
    # create character data list
    # https://youtu.be/wpyCzodvO3A?si=ACE1qNqb3_b1XETP&t=423
    # hp_characters.json was created using a helper function
    patterns: list = yt_0401.create_training_data("ner_youtube_tutorials/data/hp_characters.json", "PERSON")
    yt_0401.generate_rules(patterns)
    # print (patterns)

    nlp = spacy.load("ner_youtube_tutorials/hp_folders/hp_ner")
    ie_data = {}
    with open("ner_youtube_tutorials/data/hp.txt", "r") as f:
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

    yt_0401.save_data("ner_youtube_tutorials/data/hp_data.json", ie_data)



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
    with open("ner_youtube_tutorials/data/hp.txt", "r", encoding="utf-8") as f:
        text = f.read().split("\n\n")
        print(text)

    character_names = []
    with open("ner_youtube_tutorials/data/hp_characters.json", "r", encoding="utf-8") as f:
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
