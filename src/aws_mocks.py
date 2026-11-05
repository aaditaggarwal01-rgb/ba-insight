"""Local equivalents of the AWS services used in the BA Insight design.
These classes have the same interfaces the AWS services would expose
(S3 put/get, Comprehend sentiment, DynamoDB put/get) so the application
code does not change when the real services are used. No AWS account,
credentials or payment are needed to run them here."""
import json
import os

class S3LikeStore:
    def __init__(self, root='data'):
        self.root = root
        os.makedirs(root, exist_ok=True)
    def put(self, key, text):
        path = os.path.join(self.root, key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(text)
        return path
    def get(self, key):
        with open(os.path.join(self.root, key), encoding='utf-8') as f:
            return f.read()

class ComprehendLike:
    def __init__(self):
        from src.preprocess import sentiment_score, satisfaction_label
        self._score = sentiment_score
        self._label = satisfaction_label
    def detect_sentiment(self, text):
        compound = self._score(text)
        return {'compound': round(compound, 3), 'label': self._label(compound)}

class DynamoDBLike:
    def __init__(self, path='data/predictions.json'):
        self.path = path
        self._table = json.load(open(path)) if os.path.exists(path) else {}
    def put_item(self, key, item):
        self._table[key] = item
        json.dump(self._table, open(self.path, 'w'), indent=2)
    def get_item(self, key):
        return self._table.get(key)
