import importlib.util, sys
spec = importlib.util.find_spec("EmotionDetection")
print("spec:", spec)
if spec:
    print("origin:", spec.origin)
    print("submodule_search_locations:", spec.submodule_search_locations)
