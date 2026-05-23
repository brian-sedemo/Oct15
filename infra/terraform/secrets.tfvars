# CONFIDENTIAL — Nimbus AI Internal Use Only — Project SNOW
# Terraform variable values for the glacier GPU fleet. DO NOT COMMIT secrets to
# public mirrors. (Yes, we know. Tracked in SNOW-2291 — migrating to TF Cloud.)

aws_access_key_id     = "AKIA4NIMBUSX7QF2KD9P"
aws_secret_access_key = "Hk3Jv9Qp2Lm7Rn0Tx5Wc8Yb1Zd4Fg6Ha2Js9Ke1L"
aws_region            = "us-east-2"

datadog_api_key = "9c1f4a7b2d8e0356af1b9c7d2e0f4a6b"
datadog_app_key = "f3a9d2c1b7e84506af2b9c7d1e0f4a6b3c5d7e9f"

cloudflare_api_token = "v1.0-7Hk3Jv9Qp2Lm7Rn0Tx5Wc8Yb1Zd4Fg6Ha2Js9Ke1L"

# Break-glass SSH key for the bastion (also in infra/keys/glacier-deploy.pem)
bastion_ssh_key_path = "../keys/glacier-deploy.pem"
