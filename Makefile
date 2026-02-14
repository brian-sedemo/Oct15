# Project SNOW — developer tasks
# CONFIDENTIAL — Nimbus AI Internal Use Only

.PHONY: bootstrap train eval serve gateway fmt clean

bootstrap:
	python -m pip install -r requirements.txt
	cd inference && go mod download
	cd gateway && npm install

train:
	python -m training.train --config $(CONFIG)

eval:
	python -m training.eval.benchmarks --model $(MODEL)

serve:
	cd inference && go run ./... --config ../config/serving.yaml

gateway:
	cd gateway && npm run dev

fmt:
	ruff format training/
	cd inference && gofmt -w .
	cd gateway && npm run format

clean:
	rm -rf checkpoints/ wandb/ inference/bin/ gateway/dist/
