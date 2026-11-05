"""Model loading and prediction for BA Insight.
The trained pipeline (TF-IDF + Logistic Regression) is stored in
models/ba_satisfaction_model.joblib. In the AWS design this artefact
lives in Amazon S3 and is loaded by the prediction Lambda function."""
import os
import joblib
from src.preprocess import clean_text, sentiment_score, satisfaction_label, aspect_sentiment
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'ba_satisfaction_model.joblib')
_model = None

def load_model(path=MODEL_PATH):
    global _model
    if _model is None:
        _model = joblib.load(path)
    return _model

def predict_review(text):
    model = load_model()
    cleaned = clean_text(text)
    predicted = str(model.predict([cleaned])[0])
    compound = sentiment_score(text)
    return {
        'predicted_satisfaction': predicted,
        'vader_compound': round(compound, 3),
        'vader_label': satisfaction_label(compound),
        'aspects': aspect_sentiment(text),
    }
