"""
Emotion Detection module using Watson NLP Embedded Library.
Detects emotions (anger, disgust, fear, joy, sadness) from text input.
"""
import json
import requests


def emotion_detector(text_to_analyse):
    """
    Analyzes the given text and returns emotion scores along with the dominant emotion.

    Args:
        text_to_analyse (str): The text to analyze for emotions.

    Returns:
        dict: A dictionary containing scores for anger, disgust, fear, joy,
              sadness, and the dominant_emotion. Returns None values if input is blank.
    """
    # Handle blank/empty input - Task 7 error handling
    if not text_to_analyse or not text_to_analyse.strip():
        return {
            'anger': None,
            'disgust': None,
            'fear': None,
            'joy': None,
            'sadness': None,
            'dominant_emotion': None
        }

    url = (
        'https://sn-watson-emotion.labs.skills.network/v1/'
        'watson.runtime.nlp.v1/NlpService/EmotionPredict'
    )

    headers = {
        'grpc-metadata-mm-model-id': 'emotion_aggregated-workflow_lang_en_stock'
    }

    input_json = {
        "raw_document": {
            "text": text_to_analyse
        }
    }

    # Task 7: Handle connection/timeout errors (Watson only accessible from IBM lab)
    try:
        response = requests.post(url, headers=headers, json=input_json, timeout=10)
    except requests.exceptions.RequestException:
        return {
            'anger': None,
            'disgust': None,
            'fear': None,
            'joy': None,
            'sadness': None,
            'dominant_emotion': None
        }

    # Task 7: Handle status code 400 (bad request)
    if response.status_code == 400:
        return {
            'anger': None,
            'disgust': None,
            'fear': None,
            'joy': None,
            'sadness': None,
            'dominant_emotion': None
        }

    # Parse the response
    response_dict = json.loads(response.text)

    # Extract emotion predictions
    emotions = response_dict['emotionPredictions'][0]['emotion']

    anger_score = emotions['anger']
    disgust_score = emotions['disgust']
    fear_score = emotions['fear']
    joy_score = emotions['joy']
    sadness_score = emotions['sadness']

    # Find the dominant emotion (Task 3: format output)
    emotion_scores = {
        'anger': anger_score,
        'disgust': disgust_score,
        'fear': fear_score,
        'joy': joy_score,
        'sadness': sadness_score
    }

    dominant_emotion = max(emotion_scores, key=emotion_scores.get)

    return {
        'anger': anger_score,
        'disgust': disgust_score,
        'fear': fear_score,
        'joy': joy_score,
        'sadness': sadness_score,
        'dominant_emotion': dominant_emotion
    }