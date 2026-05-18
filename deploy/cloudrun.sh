#!/usr/bin/env bash
# One-shot Cloud Run deploy for healthcare-genai-engineer.
# Usage:  bash deploy/cloudrun.sh
#
# Prereqs (one-time per project, run from project root):
#   gcloud config set project $GCP_PROJECT
#   gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
#
# Cost: $0 idle (scale-to-zero), ~$0.40 per 1k requests at 1Gi/1CPU.

set -euo pipefail

PROJECT="${GCP_PROJECT:-bchan-genai-lab}"
REGION="${GCP_REGION:-us-west1}"
SERVICE="${SERVICE_NAME:-healthcare-genai}"
REPO="cloud-run-source-deploy"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/${SERVICE}:latest"

echo "[1/3] Building image → ${IMAGE}"
gcloud builds submit --tag "$IMAGE" --project "$PROJECT" .

echo "[2/3] Deploying to Cloud Run..."
gcloud run deploy "$SERVICE" \
    --image "$IMAGE" \
    --region "$REGION" \
    --project "$PROJECT" \
    --platform managed \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 1 \
    --timeout 300 \
    --min-instances 0 \
    --max-instances 3 \
    --set-env-vars "USE_LLM=false"

echo "[3/3] Service URL:"
gcloud run services describe "$SERVICE" \
    --region "$REGION" --project "$PROJECT" \
    --format='value(status.url)'
