# import pandas as pd
# import sklearn
# from sklearn.datasets import fetch_openml
# from keras.utils import to_categorical
# from sklearn.model_selection import train_test_split
# import numpy as np
# import json
# from matplotlib import pyplot as plt
# import ner_youtube_functions
# import tutorials
# import utility_functions as util_func
# import json
# import random
# import ner_youtube_tutorials.ner_youtube_tutorial_04_01 as yt_0401
# import ner_youtube_tutorials.ner_youtube_tutorial_04_02 as yt_0402
import ner_youtube_tutorials.ner_youtube_tutorial_04_03 as yt_0403
import re
import spacy
import time
import os
import tutorials


# NYT_PATH = 'ner_youtube_tutorials'
# NYT_DATA_PATH = os.path.join(NYT_PATH, "data")
# NYT_HP_FOLDER_PATH = os.path.join(NYT_PATH, "hp_folders")


def ner_youtube_tutorial():
    # https://youtu.be/6HM75qOsgkU?si=smI6H-dIVCQrRtW-
    # TRAIN_DATA = load_data("data/hp_training_data.json")
    # nlp = train_spacy(TRAIN_DATA, 30)
    # nlp.to_disk("hp_ner_model")

    test = "Harry James[59] Potter (b. 31 July 1980[1]) was an English half-blood[2] wizard, and one of the most famous wizards of modern times. He was the only child and son of James and Lily Potter (née Evans), both members of the original Order of the Phoenix. Harry's birth was overshadowed by a prophecy, naming either himself or Neville Longbottom as the one with the power to vanquish Lord Voldemort. After half of the prophecy was reported to Voldemort, courtesy of Severus Snape, Harry was chosen as the target due to his many similarities with the Dark Lord. In turn, this caused the Potter family to go into hiding. Voldemort made his first vain attempt to circumvent the prophecy when Harry was a year and three months old. During this attempt, he murdered Harry's parents as they tried to protect him, but this unsuccessful attempt to kill Harry led to Voldemort's first downfall. This downfall marked the end of the First Wizarding War, and to Harry henceforth being known as the 'Boy Who Lived',[5] as he was the only known survivor of the Killing Curse. One consequence of Lily's loving sacrifice was that her orphaned son had to be raised by her only remaining blood relative, his Muggle aunt, Petunia Dursley. While in her care he would be protected from Lord Voldemort, due to the Bond of Blood charm Albus Dumbledore placed upon him.[60] This powerful charm would protect him until he became of age, or no longer called his aunt's house home. Due to Petunia's resentment of her sister and her magic gifts, Harry grew up abused and neglected. On his eleventh birthday, Harry learned that he was a wizard, from Rubeus Hagrid.[61] He began attending Hogwarts School of Witchcraft and Wizardry in 1991. The Sorting Hat was initially going to Sort Harry into Slytherin House, but Harry pleaded 'not Slytherin' and the Hat heeded this plea, instead sorting the young wizard into Gryffindor House.[62] At school, Harry became best friends with Ron Weasley and Hermione Granger. He later became the youngest Quidditch Seeker in over a century and eventually the captain of the Gryffindor House Quidditch Team in his sixth year, winning two Quidditch Cups.[63] He became even better known in his early years for protecting the Philosopher's Stone from Voldemort, saving Ron's sister Ginny Weasley, solving the mystery of the Chamber of Secrets, slaying Salazar Slytherin's basilisk, and learning how to conjure a corporeal stag Patronus at the age of thirteen. In his fourth year, Harry won the Triwizard Tournament, although the competition ended with the tragic death of Cedric Diggory and the return of Lord Voldemort. During the next school year, Harry reluctantly taught and led Dumbledore's Army. He also fought in the Battle of the Department of Mysteries, during which he lost his godfather, Sirius Black."
    test = yt_0403.clean_text(test)

    people: list = []
    nlp = spacy.load("ner_youtube_tutorials/hp_folders/hp_ner_model")
    doc = nlp(test)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            people.append(ent.text)
    print(f"{len(people)} {people}")

    # nlp = spacy.load("en_core_web_sm")
    # nlp.to_disk("ner_youtube_tutorials/models/en_core_web_sm_demo")
    people: list = []
    nlp = spacy.load("en_core_web_sm")  # off the shelf spaCy model
    doc = nlp(test)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            people.append(ent.text)
    print(f"{len(people)} {people}")

    # nlp = spacy.load("en_core_web_lg")
    # nlp.to_disk("ner_youtube_tutorials/models/en_core_web_lg_demo")
    people: list = []
    nlp = spacy.load("en_core_web_lg")  # off the shelf spaCy model
    doc = nlp(test)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            people.append(ent.text)
    print(f"{len(people)} {people}")




def main():
    # tutorials.nn_from_scratch()
    # tutorials.patrick_loeber_tutorial()
    # tutorials.samson_zhang_nn()
    ner_youtube_tutorial()


if __name__ == '__main__':
    startTime = time.time()
    main()
    endTime = time.time()
    finalTime = (endTime - startTime)

    print("\nRunning Time:", "{:.2f}".format(finalTime) + " s")
