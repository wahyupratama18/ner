import spacy
from flask import Flask, request, jsonify, render_template
from transformers import pipeline

app = Flask(__name__)

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/ner', methods=['POST'])
def ner():
    data = request.get_json()
    if 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    text = data['text']
    doc = nlp(text)
    entities = [{'text': ent.text, 'label': ent.label_} for ent in doc.ents]
    
    return jsonify({'entities': entities})

@app.route('/summarizer', methods=['POST'])
def summarizer():
    data = request.get_json()

    if 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    text = data['text']
    summarizer = pipeline("summarization", model="t5-small")
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)

    return jsonify({'summary': summary[0]['summary_text']})

if __name__ == '__main__':
    app.run(debug=True)