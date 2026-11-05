"""Text cleaning, sentiment and aspect tagging for BA Insight.
Mirrors the AWS design: this module is what the data-cleaning Lambda
function runs (see lambda_handler.py). No paid AWS service is called
in this environment; Amazon Comprehend is mirrored by VADER (below)."""
import re
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
import numpy as np
STOP = set(stopwords.words('english'))
LEM = WordNetLemmatizer()
SIA = SentimentIntensityAnalyzer()
ASPECTS = {
    'Seat comfort': ['seat', 'legroom', 'leg room', 'cushion', 'cramped', 'recline', 'space'],
    'Food and drink': ['food', 'meal', 'breakfast', 'lunch', 'dinner', 'drink', 'coffee', 'wine', 'menu'],
    'Staff service': ['staff', 'crew', 'cabin crew', 'attendant', 'pilot', 'service', 'helpful', 'rude', 'friendly'],
    'In-flight entertainment': ['entertainment', 'screen', 'movie', 'film', 'wifi', 'wi-fi', 'headphone', 'music'],
    'Value for money': ['price', 'expensive', 'cheap', 'value', 'money', 'fare', 'cost', 'worth'],
    'Delays and punctuality': ['delay', 'delayed', 'late', 'cancel', 'cancelled', 'punctual', 'on time', 'waiting'],
    'Baggage': ['baggage', 'luggage', 'bag', 'suitcase', 'lost bag'],
    'Cleanliness': ['clean', 'dirty', 'toilet', 'lavatory', 'smell', 'hygiene'],
    'Booking and check-in': ['booking', 'check-in', 'check in', 'boarding', 'gate', 'ticket', 'online check'],
}

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = [LEM.lemmatize(w) for w in word_tokenize(text)
              if w.isalpha() and w not in STOP and len(w) > 2]
    return ' '.join(tokens)

def sentiment_score(text):
    return SIA.polarity_scores(text)['compound']

def satisfaction_label(compound):
    if compound >= 0.05:
        return 'Satisfied'
    if compound <= -0.05:
        return 'Dissatisfied'
    return 'Neutral'

def aspect_sentiment(text):
    result = {}
    sentences = sent_tokenize(text.lower())
    for aspect, keywords in ASPECTS.items():
        hits = [s for s in sentences if any(k in s for k in keywords)]
        if hits:
            result[aspect] = float(np.mean([sentiment_score(s) for s in hits]))
    return result
