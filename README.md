# FOD Pipeline

Foreign Object Debris (FOD) recognition pipeline built around YOLO, MobileCLIP,
and an MLP classifier. The pipeline is designed to run on AWS SageMaker using data stored in Amazon S3,
while also supporting single-image inference locally.

## Architecture

1. **Object Detection** - YOLO locates the foreign object and crops it with a 20% margin.
2. **Embedding Extraction** - MobileCLIP converts the cropped image into a 512-dimensional embedding. (Run once).
3. **MobileCLIP prediction** - The embedding is compared against text prompts to produce the Top-1 and Top-2 predictions.
4. **Classifier prediction** - The same embedding is passed through a trained MLP classifier.
5. **Hybrid prediction** - Combines MobileCLIP Top-1, MobileCLIP Top-2, and the MLP classifier prediction.

## Repository Structure

| Package | Purpose |
| :--- | :--- |
| `src/fod_pipeline/core/` | YOLO detection, MobileCLIP embedding, S3 utilities, label mapping |
| `src/fod_pipeline/data/` | FOD categories and MobileCLIP prompts |
| `src/fod_pipeline/classifier/` | MLP model, data preparation, training |
| `src/fod_pipeline/hybrid/` | Hybrid evaluation metrics |
| `src/fod_pipeline/pipeline/` | Pipeline orchestration |
| `src/fod_pipeline/sagemaker/` | SageMaker job launchers |

## Installation

> **Prerequisites:** Python `>= 3.10`

### Step 1 — Clone the Repository

```bash
git clone https://github.com/TalaChehade4/fod-pipeline.git
cd fod-pipeline
```

### Step 2 — Install sagemaker version

```bash
pip install -e ".[sagemaker]"
```

## AWS Configuration

### Step 1 — Create the environment file
Copy `.env.example` to `.env` before using S3 utilities (`core/s3_io.py`) or launching SageMaker jobs (`sagemaker/`):

```bash
cp .env.example .env
```
### Step 2 — Configure environment variables
Open `.env` and set your AWS resources:
```bash
AWS_REGION=your-aws-region
SAGEMAKER_ROLE_ARN=arn:aws:iam::123456789012:role/service-role/AmazonSageMaker-ExecutionRole
S3_BUCKET=your-s3-bucket-name
S3_PROJECT_PREFIX=your-project-prefix
```
**Note:** These variables are loaded automatically across the pipeline. You only need to pass an explicit path flag if you want to override a default S3 location.

## Running the Pipeline on SageMaker
The complete pipeline is intended to run on AWS SageMaker.

### Step 1 — Upload model weights
> **Note:** This step only needs to be run **once** before launching training or evaluation pipelines.
They are currently saved at `s3://oreyeon-models/Tala-temp/fod-pipeline/weights/`.

```bash
fod-upload yolo_model.tar.gz --kind yolo-weights
fod-upload mobileclip_s0.pt --kind mobileclip-weights
```

### Step 2 — Upload datasets
> **Note:** This step only needs to be run **once** before launching training or evaluation pipelines.
Upload the required dataset manifests, database files, and join mappings used to construct the ground truth for MobileCLIP embeddings and the MLP classifier. They are currently saved at `s3://oreyeon-models/Tala-temp/fod-pipeline/manifests/`.

```bash
# --- 1. Dataset Manifests ---
# Saved automatically as 'train_manifest.json' (Classifier training data)
fod-upload train_manifest.manifest --kind manifest --split train in S3

# Saved automatically as 'test_manifest.json' (MobileCLIP & classifier evaluation data)
fod-upload test_manifest.manifest --kind manifest --split test

# --- 2. Database CSVs ---
# Saved automatically as 'trainingdata_old.csv' (Training FOD IDs & labels of the database)
fod-upload trainingdata_old.csv --kind database-csv --split train

# Saved automatically as 'testingdata_old.csv' (Testing FOD IDs & labels of the database)
fod-upload testingdata_old.csv --kind database-csv --split test

# Saved automatically as 'join_config.json' (Classes mapping for classifier only)
fod-upload join_config.json --kind join-config
```

### Step 3 — Build label maps
> **Note:** This step only needs to be run **once** before launching training or evaluation pipelines.
Generate the required ground-truth label maps. They are currently saved at `s3://oreyeon-models/Tala-temp/fod-pipeline/manifests/`.

```bash
# Classifier ground truth (train split) - join-config checked first, database fallback. Saved automatically as `train_label_map.json`
fod-build-label-map --split train --ground-truth classifier

# Classifier ground truth (test split). Saved automatically as `test_label_map.json`
fod-build-label-map --split test --ground-truth classifier

# MobileCLIP ground truth (test split) - database only, no join-config. Saved automatically as `test_mobileclip_label_map.json`
fod-build-label-map --split test --ground-truth mobileclip
```

### Step 4 — Extract embeddings
> **Note:** This step only needs to be run **once** before launching training or evaluation pipelines unless new training or testing data is provided.
Generate MobileCLIP embeddings for the training and testing image datasets. Extracted embeddings are automatically saved in S3 at `s3://oreyeon-models/Tala-temp/fod-pipeline/embeddings/`.

```bash
# Extract embeddings for the training set
fod-sm-embed --split train

# Extract embeddings for the testing set
fod-sm-embed --split test
```

### Step 5 — Prepare classifier data
> **Note:** This step only needs to be run **once** before launching training or evaluation pipelines unless new training or testing data is provided or new classes were added.
Preprocesses the extracted embeddings to generate the following core artifacts: 

* **Train / Validation Split:** Randomly shuffles the training embeddings and splits them into **90% train** and **10% validation** sets (`train` and `val` embedding files).
* **Label Encoder:** Maps each FOD class label to a unique numerical index.
* **Class Weights:** Computes class distribution weights to handle class imbalance during loss calculation (weighted cross-entropy).

Saved automatically in S3 at `s3://oreyeon-models/Tala-temp/fod-pipeline/classifier-data/`.
```bash
fod-sm-prepare
```

# Stage 4 data prep: split + class weights + label encoding
# writes label_encoder.json to s3://<bucket>/<prefix>/classifier-data/
fod-sm-prepare

# Stage 4 training (MLP only)
fod-sm-train --epochs 100
```

SageMaker writes the raw training job output (including `model.tar.gz`)
under its own job-name/timestamp folder in `classifier-results/` - that part
can't be disabled. After the job finishes, `fod-sm-train` copies that
`model.tar.gz` to a fixed key, `classifier-models/model.tar.gz`, which is
overwritten on every run and always points at the latest trained model:

```bash
# Stage 4/5 evaluation: full hybrid metrics report against dual ground truth
# (classifier fallback is on by default here too; pass --no-classifier-fallback to disable)
fod-sm-evaluate
```

`--classifier-weights-uri`/`--label-encoder-uri` both default to the paths
above (`classifier-models/model.tar.gz`, `classifier-data/label_encoder.json`)
under `S3_PROJECT_PREFIX` - pass either explicitly only to override, e.g. to
evaluate a specific past run's artifacts instead of the latest, still
available under `classifier-results/<job-name>/output/model.tar.gz`.

Results land in `s3://<bucket>/<prefix>/hybrid-results/`
(`--output-uri` to override).

## Tests

```bash
pytest
```
# Local single-image inference (forward pass)
fod-infer path/to/image.jpg --yolo best.pt --mobileclip mobileclip_s0.pt \
    --classifier-weights FinalModel/model.pth \
    --label-encoder PreparedData/label_encoder.json
```
