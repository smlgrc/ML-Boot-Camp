import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import re

import spacy
from spacy.matcher import Matcher

from spacy import displacy
# import visualise_spacy_tree
# from IPython.display import Image, display


# load english language model
nlp = spacy.load('en_core_web_sm', disable=['ner', 'textcat'])


# function to preprocess speech
def clean(text):
    # removing paragraph numbers
    text = re.sub('[0-9]+.\t', '', str(text))
    # removing new line characters
    text = re.sub('\n ', '', str(text))
    text = re.sub('\n', ' ', str(text))
    # removing apostrophes
    text = re.sub("'s", '', str(text))
    # removing hyphens
    text = re.sub("-", ' ', str(text))
    text = re.sub("— ", '', str(text))
    # removing quotation marks
    text = re.sub('\"', '', str(text))
    # removing salutations
    text = re.sub("Mr\.", 'Mr', str(text))
    text = re.sub("Mrs\.", 'Mrs', str(text))
    # removing any reference to outside text
    text = re.sub("[\(\[].*?[\)\]]", "", str(text))

    return text


# split sentences
def sentences(text):
    # split sentences and questions
    text = re.split('[.?]', text)
    clean_sent = []
    for sent in text:
        clean_sent.append(sent)
    return clean_sent


# Function to find sentences containing PMs of India
def find_names(text):
    names = []

    # Create a spacy doc
    doc = nlp(text)

    # Define the pattern
    pattern = [{'LOWER': 'prime'},
               {'LOWER': 'minister'},
               {'POS': 'ADP', 'OP': '?'},
               {'POS': 'PROPN'}]

    # Matcher class object
    matcher = Matcher(nlp.vocab)
    matcher.add("names", [pattern])
    matches = matcher(doc)

    # Finding patterns in the text
    for i in range(0, len(matches)):
        # match: id, start, end
        token = doc[matches[i][1]:matches[i][2]]
        # append token to list
        names.append(str(token))

    # Only keep sentences containing Indian PMs
    for name in names:
        if (name.split()[2] == 'of') and (name.split()[3] != "India"):
            names.remove(name)

    return names


# Function to check if keyswords like 'programs','schemes', etc. present in sentences
def prog_sent(text):
    patterns = [r'(?i)\b' + 'plan' + r'\b',
                r'(?i)\b' + 'programme' + r'\b',
                r'(?i)\b' + 'scheme' + r'\b',
                r'(?i)\b' + 'campaign' + r'\b',
                r'(?i)\b' + 'initiative' + r'\b',
                r'(?i)\b' + 'conference' + r'\b',
                r'(?i)\b' + 'agreement' + r'\b',
                r'(?i)\b' + 'alliance' + r'\b']

    output = []
    flag = 0

    # breakpoint()
    # Look for patterns in the text
    for pat in patterns:
        if re.search(pat, text) is not None:
            flag = 1
            break
    return flag


# To extract initiatives using pattern matching
def all_schemes(text, check):
    schemes = []

    doc = nlp(text)

    # Initiatives keywords
    prog_list = ['programme', 'scheme',
                 'initiative', 'campaign',
                 'agreement', 'conference',
                 'alliance', 'plan']

    # Define pattern to match initiatives names
    pattern = [{'POS': 'DET'},
               {'POS': 'PROPN', 'DEP': 'compound'},
               {'POS': 'PROPN', 'DEP': 'compound'},
               {'POS': 'PROPN', 'OP': '?'},
               {'POS': 'PROPN', 'OP': '?'},
               {'POS': 'PROPN', 'OP': '?'},
               {'LOWER': {'IN': prog_list}, 'OP': '+'}
               ]

    if check == 0:
        # return blank list
        return schemes

    # Matcher class object
    matcher = Matcher(nlp.vocab)
    matcher.add("matching", [pattern])
    matches = matcher(doc)

    for i in range(0, len(matches)):

        # match: id, start, end
        start, end = matches[i][1], matches[i][2]

        if doc[start].pos_ == 'DET':
            start = start + 1

        # matched string
        span = str(doc[start:end])

        if (len(schemes) != 0) and (schemes[-1] in span):
            schemes[-1] = span
        else:
            schemes.append(span)

    return schemes


# rule to extract initiative name
def sent_subtree(text):
    # pattern match for schemes or initiatives
    patterns = [r'(?i)\b' + 'plan' + r'\b',
                r'(?i)\b' + 'programme' + r'\b',
                r'(?i)\b' + 'scheme' + r'\b',
                r'(?i)\b' + 'campaign' + r'\b',
                r'(?i)\b' + 'initiative' + r'\b',
                r'(?i)\b' + 'conference' + r'\b',
                r'(?i)\b' + 'agreement' + r'\b',
                r'(?i)\b' + 'alliance' + r'\b']

    schemes = []
    doc = nlp(text)
    flag = 0
    # if no initiative present in sentence
    for pat in patterns:

        if re.search(pat, text) is not None:
            flag = 1
            break

    if flag == 0:
        return schemes

    # iterating over sentence tokens
    for token in doc:

        for pat in patterns:

            # if we get a pattern match
            if re.search(pat, token.text) is not None:

                word = ''
                # iterating over token subtree
                for node in token.subtree:
                    # only extract the proper nouns
                    if (node.pos_ == 'PROPN'):
                        word += node.text + ' '

                if len(word) != 0:
                    schemes.append(word)

    return schemes


# function to check output percentage for a rule,
# this function counts how many extractions it was able to perform on df
def output_per(df, out_col):
    """
    (Pdb) df[out_col]
    0                              []
    1       [which give satisfaction]
    2            [people expect life]
    3              [We have increase]
    4              [trade show signs]
                      ...
    1628                           []
    1629              [We take goals]
    1630               [who see pain]
    1631                           []
    1632                           []
    """
    result = 0

    for out in df[out_col]:
        if len(out) != 0:
            result += 1

    per = result / len(df)
    per *= 100

    return per


def generate_df() -> pd.DataFrame:
    # Folder path
    folders = glob.glob('./UNGD/UNGDC 1970-2018/Converted sessions/Session*')

    # Dataframe
    df: pd.DataFrame = pd.DataFrame(columns=['Country', 'Speech', 'Session', 'Year'])

    # Read speeches by India
    i = 0
    for file in folders:
        glob_path: str = file + '/IND*.txt'
        speech: list = glob.glob(glob_path)  # grabs all the India speeches filepaths into a list

        with open(speech[0], encoding='utf8') as f:
            # Speech
            df.loc[i, 'Speech'] = f.read()
            # Year
            df.loc[i, 'Year'] = speech[0].split('_')[-1].split('.')[0]
            # Session
            df.loc[i, 'Session'] = speech[0].split('_')[-2]
            # Country
            df.loc[i, 'Country'] = speech[0].split('_')[0].split("\\")[-1]
            # Increment counter
            i += 1

    print(df.head())  # len(df) = 49
    # print(df.loc[0, 'Speech'])
    return df


def info_ext_1_and_2() -> pd.DataFrame:
    # # load english language model
    # nlp = spacy.load('en_core_web_sm', disable=['ner', 'textcat'])

    df: pd.DataFrame = generate_df()

    # preprocessing speeches
    df['Speech_clean'] = df['Speech'].apply(clean)

    # sentences
    df['sent'] = df['Speech_clean'].apply(sentences)

    # Create a dataframe containing sentences
    df2: pd.DataFrame = pd.DataFrame(columns=['Sent', 'Year', 'Len'])

    # List of sentences for new df
    row_list = []

    # for-loop to go over the df speeches
    for i in range(len(df)):

        # for-loop to go over the sentences in the speech
        for sent in df.loc[i, 'sent']:
            wordcount = len(sent.split())  # Word count
            year = df.loc[i, 'Year']  # Year
            dict1 = {'Year': year, 'Sent': sent, 'Len': wordcount}  # Dictionary
            row_list.append(dict1)  # Append dictionary to list

    # Create the new df
    df2 = pd.DataFrame(row_list)

    print(df2.head())
    print(df2.shape)

    # # load english language model
    # nlp = spacy.load('en_core_web_sm', disable=['ner', 'textcat'])

    # Apply function
    # breakpoint()
    df2['PM_Names'] = df2['Sent'].apply(find_names)

    # look at sentences for a specific year
    for i in range(len(df2)):
        if df2.loc[i, 'Year'] in ['1984']:
            if len(df2.loc[i, 'PM_Names']) != 0:
                print('->', df2.loc[i, 'Sent'], '\n')

    # 58 sentences out of 7150 total sentences gave an output that matched our pattern
    count = 0
    for i in range(len(df2)):
        if len(df2.loc[i, 'PM_Names']) != 0:
            count += 1
    print(f'sentences that matched the pattern = {count}')

    # Apply function
    df2['Check_Schemes'] = df2['Sent'].apply(prog_sent)

    # Sentences that contain the initiative words
    count = 0
    for i in range(len(df2)):
        if df2.loc[i, 'Check_Schemes'] == 1:
            count += 1
    print(f'Sentences that contain the initiative words = {count}')

    # apply function
    df2['Schemes1'] = df2.apply(lambda x: all_schemes(x.Sent, x.Check_Schemes), axis=1)

    # how many of the sentences contain an initiative name
    count = 0
    for i in range(len(df2)):
        if len(df2.loc[i, 'Schemes1']) != 0:
            count += 1
    print(count)

    year = '2018'
    for i in range(len(df2)):
        if df2.loc[i, 'Year'] == year:
            if len(df2.loc[i, 'Schemes1']) != 0:
                print('->', df2.loc[i, 'Year'], ',', df2.loc[i, 'Schemes1'], ':')
                print(df2.loc[i, 'Sent'])

    # derive initiatives
    df2['Schemes2'] = df2['Sent'].apply(sent_subtree)

    count = 0
    for i in range(len(df2)):
        if len(df2.loc[i, 'Schemes2']) != 0:
            count += 1
    print(count)

    year = '2018'
    for i in range(len(df2)):
        if df2.loc[i, 'Year'] == year:
            if len(df2.loc[i, 'Schemes2']) != 0:
                print('->', df2.loc[i, 'Year'], ',', df2.loc[i, 'Schemes2'], ':')
                print(df2.loc[i, 'Sent'])

    row_list = []
    # df2 contains all sentences from all speeches
    for i in range(len(df2)):
        sent = df2.loc[i, 'Sent']

        if (',' not in sent) and (len(sent.split()) <= 15):
            year = df2.loc[i, 'Year']
            length = len(sent.split())

            dict1 = {'Year': year, 'Sent': sent, 'Len': length}
            row_list.append(dict1)

    # df with shorter sentences
    df3 = pd.DataFrame(columns=['Year', 'Sent', "Len"])
    df3 = pd.DataFrame(row_list)
    print(df3.head())
    return df3


# function for rule 1: noun(subject), verb, noun(object)
def rule1(text):
    doc = nlp(text)  # There are many developments in India which give us satisfaction
    sent = []
    for token in doc:
        # if the token is a verb
        if token.pos_ == 'VERB':
            phrase = ''
            # only extract noun or pronoun subjects
            for sub_tok in token.lefts:
                if (sub_tok.dep_ in ['nsubj', 'nsubjpass']) and (sub_tok.pos_ in ['NOUN', 'PROPN', 'PRON']):
                    # breakpoint()
                    # add subject to the phrase
                    phrase += sub_tok.text  # which

                    # save the root of the verb in phrase
                    phrase += ' ' + token.lemma_  # give

                    # check for noun or pronoun direct objects
                    for sub_tok in token.rights:
                        # save the object in the phrase
                        if (sub_tok.dep_ in ['dobj']) and (sub_tok.pos_ in ['NOUN', 'PROPN']):
                            phrase += ' ' + sub_tok.text  # satisfaction
                            sent.append(phrase)  # ['which give satisfaction']
        # if sent:
        #     breakpoint()
    return sent


def info_ext_3(df3: pd.DataFrame):
    # Create a df containing sentence and its output for rule 1
    row_list = []

    for i in range(len(df3)):
        sent = df3.loc[i, 'Sent']
        year = df3.loc[i, 'Year']
        output = rule1(sent)
        dict1 = {'Year': year, 'Sent': sent, 'Output': output}
        row_list.append(dict1)

    df_rule1 = pd.DataFrame(row_list)

    # Rule 1 achieves 20% result on simple sentences
    print(output_per(df_rule1, 'Output'))


def main():
    info_ext_3(info_ext_1_and_2())


if __name__ == '__main__':
    main()
