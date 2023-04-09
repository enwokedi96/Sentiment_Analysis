from flask import Flask, render_template, request
import pickle
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

# Load the trained model and vectorizer
filedir = ""
clf = pickle.load(open(filedir+"NaiiveBayes.sav", 'rb'))
vectorizer = pickle.load(open(filedir+'vectorizer.sav', 'rb')) # CountVectorizer(vocabulary=pickle.load(open(filedir+'vectorizer.sav', 'rb')))

@app.route('/')
def home():
    return render_template(r'home.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    text = vectorizer.transform([text])
    prediction = clf.predict(text)[0]

    if prediction == 1:
        result = 'Positive'
    else:
        result = 'Negative'

    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
