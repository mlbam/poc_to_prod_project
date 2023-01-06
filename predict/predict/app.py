from flask import Flask, request, render_template
import json
from run import TextPredictionModel

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route("/", methods=['POST'])
def get_prediction():
    model = TextPredictionModel.from_artefacts("C:/Users/Myra-Louise/Downloads/Capstone-20221116/poc-to-prod-capstone/poc-to-prod-capstone/train/data/artefacts/2023-01-04-15-31-27")
    text = request.form['text']
    prediction = model.predict(text)
    # return str(prediction)
    return str(prediction)


if __name__ == '__main__':
    app.run()
    # debug mode activated
    debug = True