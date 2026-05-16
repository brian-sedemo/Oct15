#!/usr/bin/env bash
# CONFIDENTIAL — Nimbus AI Internal Use Only — Project SNOW
# Deploys the Snowfall serving stack to glacier-serve-use2.
set -euo pipefail

CLUSTER="${CLUSTER:-glacier-serve-use2}"
IMAGE_TAG="${IMAGE_TAG:-r12}"
REGISTRY="registry.internal.nimbus.ai/snow"

echo "❄️  Deploying Snowfall serving (tag=${IMAGE_TAG}) to ${CLUSTER}"
echo "    CONFIDENTIAL — authorized FMG personnel only"

# Auth to the internal registry (creds from vault in CI; .env locally)
source ./.env 2>/dev/null || true

docker build -t "${REGISTRY}/inference:${IMAGE_TAG}" ./inference
docker push "${REGISTRY}/inference:${IMAGE_TAG}"

kubectl --context "${CLUSTER}" apply -f infra/k8s/deployment.yaml
kubectl --context "${CLUSTER}" -n snow-serving rollout status deploy/snowfall-serving

echo "✅ Snowfall ${IMAGE_TAG} live on ${CLUSTER}"
