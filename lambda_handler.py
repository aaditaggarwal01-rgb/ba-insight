"""Serverless prediction handler for BA Insight.
In the AWS design this is the AWS Lambda function behind Amazon API
Gateway. It is written as a plain Python module so it can be tested
locally without AWS, and deployed unchanged to Lambda later."""
import json
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.model import predict_review
from src.recommend import recommend

def handler(event, context=None):
    body = event.get('body', event) if isinstance(event, dict) else {}
    if isinstance(body, str):
        body = json.loads(body)
    text = body.get('review_text', '')
    if not text or len(text.strip()) < 3:
        return {'statusCode': 400, 'body': json.dumps({'error': 'review_text is required'})}
    result = predict_review(text)
    result['recommendation'] = recommend(result)
    return {'statusCode': 200, 'body': json.dumps(result)}

if __name__ == '__main__':
    print(json.dumps(handler({'review_text': 'The crew were wonderful and the flight was on time.'}), indent=2))
