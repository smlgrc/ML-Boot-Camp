import re

import uvicorn
from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
# from openie import StanfordOpenIE
import json
import html


""" returns html request allowing user to keep entering a string """
app = FastAPI()

# Create templates directory in the same directory as this script
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
@app.post("/", response_class=HTMLResponse)
async def process_request(request: Request, sentence: str = Form(None)):
    if request.method == "GET":
        # Handle GET request (display a form to enter a sentence)
        return templates.TemplateResponse("index2.html", {"request": request, "message": "Please enter a sentence."})
    elif request.method == "POST":
        # Handle POST request (process the submitted sentence)
        if sentence:
            # Process the sentence here (replace with your logic)
            # result = f"You entered: {sentence}"
            result = json.dumps(process_sentence_AllenNLP(sentence))
            # breakpoint()

            return templates.TemplateResponse("index2.html", {"request": request, "result": result, "message": ""})
        else:
            return templates.TemplateResponse("index2.html", {"request": request, "message": "No sentence provided in the POST request."})


# def process_sentence_coreNLP(sentence: str):
#     # test_string = "`` It 's going to be a tough league , '' promises the 47 - year - old Mr. Campaneris ."
#     subjects, predicates, objects = [], [], []
#
#     # https://stanfordnlp.github.io/CoreNLP/openie.html#api
#     # Default value of openie.affinity_probability_cap was 1/3.
#     properties = {
#         'openie.affinity_probability_cap': 2 / 3,
#     }
#
#     with StanfordOpenIE(properties=properties) as client:
#         print('Text: %s.' % sentence)
#         # Returns full or simpler format of triples <subject, relation, object>.
#         for triple in client.annotate(sentence):
#             print('|-', triple)
#             subjects.append(triple['subject'])
#             predicates.append(triple['relation'])
#             objects.append(triple['object'])
#
#     s = sentence.replace("'", "`")
#     return {
#         "sentence": html.escape(s),
#         "subjects": subjects,
#         "predicates": predicates,
#         "objects": objects
#     }


def process_sentence_AllenNLP(sentence: str):
    from allennlp.predictors.predictor import Predictor

    # Load the OpenIE predictor
    predictor = Predictor.from_path(
        "https://storage.googleapis.com/allennlp-public-models/openie-model.2020.03.26.tar.gz")

    # Extract relations and arguments
    output = predictor.predict(sentence=sentence)

    # Extract subjects, relations, and objects from the descriptions
    json_list = []
    for verb_info in output['verbs']:
        arguments = []

        description = verb_info['description']
        verb = verb_info['verb']

        # Use regular expressions to find all ARGs with numbers attached and their values
        arg_pattern = r'\[ARG(\d+): (.*?)\]'
        arg_matches = re.findall(arg_pattern, description)

        # Create a dictionary to store the extracted arguments and values
        arg_dict = {arg_num: arg_value for arg_num, arg_value in arg_matches}

        # Sort the dictionary by keys
        arg_dict = dict(sorted(arg_dict.items()))

        # Print the extracted argument and value pairs
        if verb:
            arguments.append(verb)  # adding verb to arguments because it's added to text.oie
        for arg_num, arg_value in arg_dict.items():
            if arg_num.isdigit():
                arguments.append(arg_value)

        mod_sentence = sentence.replace("'", "`")
        mod_arguments = [html.escape(arg.replace("'", "`")) for arg in arguments]
        print(f"\nSentence: {sentence}")
        print(f"Predicate: {verb}")
        print(f"Arguments: {arguments}")
        print(f"Modified Sentence: {mod_sentence}")
        json_list.append({
            "sentence": html.escape(mod_sentence),
            "predicate": verb,
            "arguments": mod_arguments
        })
    return json_list


if __name__ == "__main__":
    # uvicorn.run(debug=True)
    uvicorn.run(app, port=80, host="localhost")  # 127.0.0.1


""" returns a simple dictionary (json) """
# app = FastAPI()
#
# # Create templates directory in the same directory as this script
# templates = Jinja2Templates(directory="templates")
#
# @app.get("/")
# def read_form(request: Request):
#     # if request.method == "POST":
#     request.query_params.get("sentence")
#     return templates.TemplateResponse("index2.html", {"request": request})
#
# @app.post("/")
# def process_sentence(sentence: str = Form(...)):
#     # Process the sentence here (replace with your logic)
#     # result = f"{sentence}"
#     return {"You entered": sentence}
