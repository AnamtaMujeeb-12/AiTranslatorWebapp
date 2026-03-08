from flask import Flask, render_template, request
from transformers import MarianMTModel, MarianTokenizer

app = Flask(__name__)

model_name = "Helsinki-NLP/opus-mt-en-hi"
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)

@app.route("/", methods=["GET", "POST"])
def translate():
    translated_text = ""

    if request.method == "POST":
        text = request.form["data"]

        input_tokens = tokenizer(text, return_tensors="pt", padding=True)
        output_tokens = model.generate(**input_tokens)
        translated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

    return render_template("index.html", translated_text=translated_text)

if __name__ == "__main__":
    app.run(debug=True)