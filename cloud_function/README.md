# Cloud Function / Cloud Run entry-point for the Emotion Detector
# Deploys the same analyzer used by the Flask app, but as a serverless
# function that auto-scales and runs in <100ms.
#
# Deploy to Google Cloud Functions:
#   gcloud functions deploy emotion-detector \
#       --runtime python311 \
#       --trigger-http \
#       --allow-unauthenticated \
#       --memory 256MB \
#       --timeout 10s \
#       --entry-point analyze \
#       --source .
#
# Or to Cloud Run:
#   gcloud run deploy emotion-detector \
#       --source . --region us-central1 \
#       --allow-unauthenticated --memory 256Mi --cpu 1
#
# The function expects: { "text": "..." }
# and returns the rich JSON payload rendered by the dashboard UI.
