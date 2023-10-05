import uvicorn
from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from openie import StanfordOpenIE
import json
import html

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
            result = json.dumps(process_sentence(sentence))
            # breakpoint()

            return templates.TemplateResponse("index2.html", {"request": request, "result": result, "message": ""})
        else:
            return templates.TemplateResponse("index2.html", {"request": request, "message": "No sentence provided in the POST request."})


def process_sentence(sentence: str):
    # test_string = "`` It 's going to be a tough league , '' promises the 47 - year - old Mr. Campaneris ."
    subjects, predicates, objects = [], [], []

    # https://stanfordnlp.github.io/CoreNLP/openie.html#api
    # Default value of openie.affinity_probability_cap was 1/3.
    properties = {
        'openie.affinity_probability_cap': 2 / 3,
    }

    with StanfordOpenIE(properties=properties) as client:
        print('Text: %s.' % sentence)
        # Returns full or simpler format of triples <subject, relation, object>.
        for triple in client.annotate(sentence):
            print('|-', triple)
            subjects.append(triple['subject'])
            predicates.append(triple['relation'])
            objects.append(triple['object'])

        # sentence = html.escape(sentence.replace("'", "`"))
        # sentence = sentence.replace('&#x27;', '&apos;')
        # sentence = html.unescape(sentence)


        # s = html.unescape("hello")
        """"""
        # with open(corpus_path, encoding='utf8') as r:
        #     corpus = r.read().replace('\n', ' ').replace('\r', '')
        #     breakpoint()
        # triples_corpus = client.annotate(corpus[0:5000])
        # print('Corpus: %s [...].' % corpus[0:80])
        # print('Found %s triples in the corpus.' % len(triples_corpus))
        # for triple in triples_corpus[:3]:
        #     print('|-', triple)
        # print('[...]')
    # breakpoint()
    s = sentence.replace("'", "`")
    return {
        "sentence": html.escape(s),
        "subjects": subjects,
        "predicates": predicates,
        "objects": objects
    }


if __name__ == "__main__":
    # uvicorn.run(debug=True)
    uvicorn.run(app, port=80, host="localhost")  # 127.0.0.1



# app = FastAPI()
#
# # Create templates directory in the same directory as this script
# templates = Jinja2Templates(directory="templates")
#
# @app.get("/", response_class=HTMLResponse)
# def read_form(request: Request):
#     result = ""  # Initialize the result variable
#     return templates.TemplateResponse("index2.html", {"request": request, "result": result})
#
# @app.post("/", response_class=HTMLResponse)
# def process_sentence(request: Request, sentence: str = Form(...)):
#     # Process the sentence here (replace with your logic)
#     result = f"You entered: {sentence}"
#     return templates.TemplateResponse("index2.html", {"request": request, "result": result})


# app = FastAPI()
# templates = Jinja2Templates(directory="templates")
#
# @app.get("/")
# def read_root(request: Request):
#     return templates.TemplateResponse("index2.html", {"request": request})
#
# @app.post("/process/")
# async def process_sentence(sentence: str = Form(...)):
#     # Perform some processing on the sentence (e.g., reverse it)
#     processed_sentence = sentence[::-1]
#     return {"processed_sentence": processed_sentence}