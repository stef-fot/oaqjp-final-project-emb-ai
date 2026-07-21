from EmotionDetection.emotion_detection import emotion_detector

samples = [
    "i want my ex back",
    "μου λείπει ο/η πρώην μου",
    "I just got promoted today!",
    "Estoy muy enojado contigo",
    "今日とても嬉しいです",
    "asdf qwer zxcv",
]

for t in samples:
    r = emotion_detector(t)
    if r.get("ok"):
        print(
            f"OK   | {t!r:42} -> lang={r['language']:14} primary={r['primary_emotion']:9}"
            f" positivity={r['sentiment']['positivity']:5.1f}%"
            f" conf={r['confidence']:.2f} in {r['elapsed_ms']:.1f}ms"
        )
    else:
        print(f"INFO | {t!r:42} -> {r['message']}")
