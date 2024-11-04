import pickle
import torch
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load the CountVectorizer
with open('count_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

def load_model_on_cpu(file_path):
    """Loads a model from a given file path and ensures it's on CPU."""
    try:
        # Load the model with map_location set to 'cpu' for safety.
        with open(file_path, 'rb') as model_file:
            loaded_model = pickle.load(model_file)
            
            # If it's a PyTorch-like object (though unlikely for sk-learn),
            # you'd move it here explicitly.
            return loaded_model.to(device=torch.device("cpu")) if hasattr(loaded_model, 'to') else loaded_model
   
    except Exception as e:
        print(f"Error loading model from {file_path}: {str(e)}")
        raise

# Load logistic regression model into CPU memory.
model = load_model_on_cpu('logistic_regression_model.pkl')

def predict_sentiment(text):
    # Transform the input text using the CountVectorizer
    input_features = vectorizer.transform([text])

    # Make prediction
    prediction = model.predict(input_features)[0]

    return prediction

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def result():
    # Check if 'text' key exists in the form data or query parameters
    text = request.form.get('text') or request.json.get('text') if request.method == 'POST' else request.args.get('text')
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    prediction = predict_sentiment(text)
    sentiment = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}.get(prediction, 'Unknown')

    # Debug statement to print the prediction and sentiment
    print("Prediction:", prediction)
    print("Sentiment:", sentiment)

    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)