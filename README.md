# fod-pipeline

FOD (foreign object debris) recognition pipeline: YOLO detection + MobileCLIP
embedding + MLP classifier, combined into a hybrid prediction with dual
ground-truth evaluation.

## Architecture

1. **Detect** - YOLO locates the object, crops it with a 20% expansion margin.
2. **Embed** - MobileCLIP encodes the crop into a 512-D embedding (run once,
   reused by both of the next two steps).
3. **MobileCLIP prediction** - Top-1/Top-2 categories via text-prompt similarity.
4. **Classifier prediction** - the same embedding, fed through an MLP2 classifier.
5. **Hybrid prediction** - the 3 candidates above; an image counts as correctly
   recognized if either model matches its own ground truth (OR rule).

Each stage's code lives under `src/fod_pipeline/`:

| Package | Purpose |
|---|---|
| `core/` | YOLO detection, MobileCLIP embedding, S3 I/O, label mapping |
| `classifier/` | MLP2 model, data prep, training, evaluation |
| `hybrid/` | dual ground truth + OR-rule metrics |
| `pipeline/` | orchestration: `preprocess` (Stage 1+2), `infer` (local forward pass), `evaluate` (Stage 4/5 hybrid report) |
| `sagemaker/` | job launch scripts for running the above on AWS SageMaker |

## Install

```bash
pip install -e ".[dev]"
```

Copy `.env.example` to `.env` and fill in your AWS values before using anything
under `sagemaker/` (launching a job) or `core/s3_io.py` (reading from S3):

```bash
cp .env.example .env
```

## CLI usage

```bash
# Stage 1+2: extract embeddings from a manifest
fod-preprocess --manifest manifest.json --label-map label_map.json \
    --yolo best.pt --mobileclip mobileclip_s0.pt --output-dir embeddings/

# Stage 4 data prep: split + class weights + label encoding
fod-prepare --input-dir embeddings/ --output-dir PreparedData/

# Stage 4 training (MLP2 only)
fod-train --train-dir PreparedData/ --model-dir FinalModel/

# Stage 4/5 evaluation: full hybrid metrics report against dual ground truth
fod-evaluate --manifest test_manifest.json \
    --mobileclip-label-map ground_truth_a.json \
    --classifier-label-map ground_truth_b.json \
    --yolo best.pt --mobileclip mobileclip_s0.pt \
    --classifier-weights FinalModel/model.pth \
    --label-encoder PreparedData/label_encoder.json \
    --output-dir evaluation-results/

# Local single-image inference (forward pass)
fod-infer path/to/image.jpg --yolo best.pt --mobileclip mobileclip_s0.pt \
    --classifier-weights FinalModel/model.pth \
    --label-encoder PreparedData/label_encoder.json
```

To run the same stages on SageMaker instead, see the `launch_*.py` scripts
under `src/fod_pipeline/sagemaker/` (requires `pip install -e ".[sagemaker]"`).

## Tests

```bash
pytest
```
