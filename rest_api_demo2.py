from flask import Flask, render_template, request

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    result = "Please enter a sentence."

    if request.method == "POST":
        sentence = request.form.get("sentence")
        # Perform some processing or analysis on the sentence here
        result = process_sentence(sentence)

    # return result
    return render_template("index.html", result=result)


def process_sentence(sentence):
    # Replace this with your sentence processing logic
    # return "You entered: " + sentence
    return {
        "You entered ": sentence
    }


if __name__ == "__main__":
    app.run(debug=True)
