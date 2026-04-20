"""
Flask server for the Emotion Detection web application.
Provides web interface and API endpoint for emotion analysis.
"""

from flask import Flask, render_template, request
from EmotionDetection.emotion_detection import emotion_detector

app = Flask("Emotion Detector")


@app.route("/emotionDetector")
def sent_detector():
    """
    Analyzes emotion from a text query parameter.

    Returns:
        str: Formatted emotion analysis result or error message.
    """
    text_to_analyse = request.args.get('textToAnalyse')

    # Task 7: Handle blank input error
    if not text_to_analyse or not text_to_analyse.strip():
        return "Invalid text! Please try again."

    result = emotion_detector(text_to_analyse)

    # Task 7: Handle None dominant emotion (from status 400)
    if result['dominant_emotion'] is None:
        return "Invalid text! Please try again."

    # Task 3: Format the output
    response = (
        f"For the given statement, the system response is "
        f"'anger': {result['anger']}, "
        f"'disgust': {result['disgust']}, "
        f"'fear': {result['fear']}, "
        f"'joy': {result['joy']} and "
        f"'sadness': {result['sadness']}. "
        f"The dominant emotion is <b>{result['dominant_emotion']}</b>."
    )

    return response


@app.route("/")
def render_index_page():
    """Renders the main index page."""
    return render_template('index.html')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
