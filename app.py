import nltk
from flask import Flask, request, jsonify, render_template
from transformers import pipeline

app = Flask(__name__)

# Download the necessary NLTK models
nltk.download('punkt')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')

# Load the summarization pipeline
# summarizer = pipeline("summarization", model="t5-base", tokenizer="t5-base")

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# @app.route('/summarize', methods=['POST'])
# def summarize():
#     data = request.get_json()
#     if 'text' not in data:
#         return jsonify({'error': 'No text provided'}), 400
    
#     text = data['text']
#     summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
    
#     return jsonify({'summary': summary[0]['summary_text']})

@app.route('/ner', methods=['POST'])
def ner():
    data = request.get_json()
    if 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    text = data['text']
    sentences = nltk.sent_tokenize(text)
    tokens = [nltk.word_tokenize(sentence) for sentence in sentences]
    pos_tags = [nltk.pos_tag(token) for token in tokens]
    chunks = [nltk.ne_chunk(pos_tag) for pos_tag in pos_tags]
    
    entities = []
    for chunk in chunks:
        for subtree in chunk:
            if isinstance(subtree, nltk.Tree):
                entity = " ".join([token for token, pos in subtree.leaves()])
                label = subtree.label()
                entities.append({'text': entity, 'label': label})
    
    return jsonify({'entities': entities})

if __name__ == '__main__':
    app.run(debug=True)