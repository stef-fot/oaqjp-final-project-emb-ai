# Emotion Detection - AI-Based Web Application

## Project Overview
An AI-powered web application that performs emotion detection on customer feedback
using the Watson NLP Embedded Library. The system identifies emotions including
**anger**, **disgust**, **fear**, **joy**, and **sadness** from text input.

## Project Structure
```
EmotionDetection/
├── EmotionDetection/
│   ├── __init__.py               # Package initializer
│   └── emotion_detection.py      # Core Watson NLP logic
├── templates/
│   └── index.html                # Web UI
├── server.py                     # Flask web server
├── test_emotion_detection.py     # Unit tests
└── README.md
```

## Tasks Completed
- ✅ Task 1: Forked and cloned the project repository
- ✅ Task 2: Created emotion detection application using Watson NLP library
- ✅ Task 3: Formatted the output of the application
- ✅ Task 4: Packaged the application (`__init__.py`)
- ✅ Task 5: Unit tests with `unittest`
- ✅ Task 6: Deployed as web application using Flask
- ✅ Task 7: Incorporated error handling (blank input & status 400)
- ✅ Task 8: Static code analysis with PyLint (10/10 score)

## Setup & Installation

```bash
pip install flask requests
```

## Running the Application

```bash
cd EmotionDetection
python server.py
```

Then open: http://localhost:5000

## Running Unit Tests

```bash
python -m pytest test_emotion_detection.py -v
```

## Running Static Code Analysis

```bash
pylint server.py EmotionDetection/emotion_detection.py
```

## API Endpoint

```
GET /emotionDetector?textToAnalyse=<your text>
```

**Example Response:**
```
For the given statement, the system response is 'anger': 0.006, 'disgust': 0.002,
'fear': 0.009, 'joy': 0.976 and 'sadness': 0.005. The dominant emotion is joy.
```

## Error Handling
- Blank input returns: `"Invalid text! Please try again."`
- API status 400 returns: `"Invalid text! Please try again."`
