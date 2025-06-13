from flask import Flask, request, render_template_string
import joblib
import re
import os
import subprocess

#Run analysis2.py only if models don't exist
if not all([os.path.exists(p) for p in ["sentiment_model.joblib", "tfidf_vectorizer.joblib", "label_encoder.joblib"]]):
    subprocess.run(["python", "analysis2.py"])

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("sentiment_model.joblib")
vectorizer = joblib.load("tfidf_vectorizer.joblib")
label_encoder = joblib.load("label_encoder.joblib")

# HTML template
HTML_PAGE = """
<!doctype html>
<title>Sentiment Analysis</title>
<h2>Enter text to analyze sentiment</h2>
<form method=post action="/predict">
  <textarea name=text rows="5" cols="50"></textarea><br><br>
  <input type=submit value="Sentiment Analyze">
</form>
{% if sentiment %}
  <h3>Predicted Sentiment: {{ sentiment }}</h3>
{% endif %}
"""

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML_PAGE)

@app.route('/predict', methods=['POST'])
def predict():
    raw_text = request.form.get("text", "")
    cleaned = clean_text(raw_text)
    vect = vectorizer.transform([cleaned])
    pred = model.predict(vect)
    label = label_encoder.inverse_transform(pred)[0]
    return render_template_string(HTML_PAGE, sentiment=label)

if __name__ == '__main__':
    app.run(debug=True)
