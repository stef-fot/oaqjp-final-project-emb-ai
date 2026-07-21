"""
EmotionDetection package entry point.

Exposes the multilingual emotion detector so callers can do:
    from EmotionDetection import emotion_detector
"""

from EmotionDetection.emotion_detection import emotion_detector

__all__ = ["emotion_detector"]
__version__ = "2.0.0"
