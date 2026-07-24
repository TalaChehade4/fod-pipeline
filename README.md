# FOD Pipeline

FOD (foreign object debris) recognition pipeline: YOLO detection + MobileCLIP
embedding and similarity matching + MLP classifier, combined into a hybrid prediction with dual
ground-truth evaluation.

## Architecture

1. **Detect** - YOLO locates the object, crops it with a 20% expansion margin.
2. **Embed** - MobileCLIP encodes the crop into a 512-D embedding (run once,
   reused by both of the next two steps).
3. **MobileCLIP prediction** - Top-1/Top-2 categories via text-prompt similarity.
4. **Classifier prediction** - the same embedding, fed through an MLP classifier.
5. **Hybrid prediction** - combines MobileCLIP Top-1, MobileCLIP Top-2, and the MLP classifier prediction.
   An image is considered correctly recognized if either model matches its corresponding ground truth.

Each stage's code lives under `src/fod_pipeline/`:

| Package | Purpose |
|---------|---------|
| `core/` | YOLO detection, MobileCLIP embedding, S3 I/O, label mapping |
| `data/` | Contains the FOD categories and prompts used by MobileCLIP|
| `classifier/` | MLP model, data prep, training, evaluation |
| `hybrid/` | dual ground truth + OR-rule metrics |
| `pipeline/` | orchestration: `preprocess` (Stage 1+2), `infer` (local forward pass), `evaluate` (Stage 4/5 hybrid report) |
| `sagemaker/` | job launch scripts for running the above on AWS SageMaker |


## Clone repository

```bash
git clone https://github.com/TalaChehade4/fod-pipeline.git
cd fod-pipeline
```

## Install locally

```bash
git clone https://github.com/TalaChehade4/fod-pipeline.git
cd fod-pipeline
pip install -e ".[dev]"
```
## Install for sagemaker

```bash
pip install -e ".[sagemaker]"
```

Requires Python >= 3.10.

Copy `.env.example` to `.env` and fill in your AWS values before using anything
under `sagemaker/` (launching a job) or `core/s3_io.py` (reading from S3):

```bash
cp .env.example .env
```

## Local CLI usage

Everything here runs on your machine against local files - no S3, no
SageMaker. Skip to **Running entirely on SageMaker** below if that's your
workflow instead.

```bash
# Stage 0: build label_map.json from a manifest + a lookup source. Needed twice,
# once per ground truth - label_map.json is always a consumed input everywhere else,
# nothing else produces it:
#   - classifier ground truth: --join-config (curated overrides) checked first,
#     --csv (e.g. trainingdata_old.csv/testingdata_old.csv) fills the rest
#   - MobileCLIP ground truth: --csv only, no --join-config
fod-build-label-map --manifest train_manifest.json --csv trainingdata_old.csv \
    --id-column trainingID --join-config join_config.json --output classifier_label_map.json
fod-build-label-map --manifest train_manifest.json --csv trainingdata_old.csv \
    --id-column trainingID --output mobileclip_label_map.json

# Stage 1+2: extract embeddings from a manifest
fod-preprocess --manifest manifest.json --label-map label_map.json \
    --yolo best.pt --mobileclip mobileclip_s0.pt --output-dir embeddings/

# Stage 4 data prep: split + class weights + label encoding
fod-prepare --input-dir embeddings/ --output-dir PreparedData/

# Stage 4 training (MLP only)
fod-train --train-dir PreparedData/ --model-dir FinalModel/

# Stage 4/5 evaluation: full hybrid metrics report against dual ground truth
# (classifier fallback for MobileCLIP's Top-1/Top-2 matching is on by default;
# pass --no-classifier-fallback for strict Ground-Truth-A-only matching)
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

## Running entirely on SageMaker (S3-backed)

Use this section if all your data lives in S3 and you never want to touch a
local copy of a manifest, CSV, or model file. Every command below either
reads/writes S3 directly or launches a SageMaker job that reads/writes S3 -
nothing is expected to exist on your laptop's filesystem. Requires
`pip install -e ".[sagemaker]"`.

Every S3 location involved - weights, manifests, database CSVs, join-config,
label maps, and the `-uri` flags on the `fod-sm-*` commands - defaults to a
path computed from `S3_BUCKET` + `S3_PROJECT_PREFIX` in `.env`, so once
`.env` is filled in and step 3's files are uploaded once, none of the
commands below need an S3 URI typed by hand. Pass a flag explicitly only to
override one location - e.g. if a file lives somewhere other than its
default path.

(If you ever do need to write a URI by hand, `<bucket>`/`<prefix>` mean
"substitute your real `S3_BUCKET`/`S3_PROJECT_PREFIX` value here", not a
shell variable. `.env` is only loaded by `python-dotenv` inside the Python
process - it is never exported to your terminal, so typing the literal
`$S3_BUCKET` or `%S3_BUCKET%` will not expand to anything.)

### 1. Configure `.env`

```bash
cp .env.example .env
```

Fill in `AWS_REGION`, `SAGEMAKER_ROLE_ARN`, `S3_BUCKET`, `S3_PROJECT_PREFIX`.

### 2. Upload model weights (one-time)

```bash
fod-upload yolo_model.tar.gz --kind yolo-weights
fod-upload mobileclip_s0.pt --kind mobileclip-weights
```

### 3. Get your manifests, database CSVs, and join-config into S3 (one-time)

```bash
fod-upload train_manifest.manifest --kind manifest --split train
fod-upload test_manifest.manifest --kind manifest --split test

fod-upload trainingdata_old.csv --kind database-csv --split train
fod-upload testingdata_old.csv --kind database-csv --split test
fod-upload join_config.json --kind join-config
```

### 4. Build the label maps

Once step 3's files are in place, every input defaults - `fod-build-label-map`
needs nothing but `--split` (+ `--ground-truth` to pick which one):

```bash
# Classifier ground truth (train split) - join-config checked first, CSV fallback
fod-build-label-map --split train --ground-truth classifier

# Classifier ground truth (Ground Truth B, test split)
fod-build-label-map --split test --ground-truth classifier

# MobileCLIP ground truth (Ground Truth A, test split) - CSV only, no join-config
fod-build-label-map --split test --ground-truth mobileclip
```

(`--csv`/`--id-column`/`--join-config`/`--manifest`/`--output` still accept
explicit local paths or `s3://` URIs if your files live somewhere other than
the step-3 defaults - `fod-build-label-map` downloads/uploads either way, so
nothing has to touch a persistent local file regardless.)

### 5. Run the pipeline stages

```bash
# Stage 1+2: extract embeddings (train split, then test split)
fod-sm-embed --split train
fod-sm-embed --split test

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
fod-sm-evaluate \
    --classifier-weights-uri s3://<bucket>/<prefix>/classifier-models/model.tar.gz \
    --label-encoder-uri s3://<bucket>/<prefix>/classifier-data/label_encoder.json
```

If you need the artifacts from a specific past run rather than the latest,
they're still there under `classifier-results/<job-name>/output/model.tar.gz`.

Results land in `s3://<bucket>/<prefix>/hybrid-results/`
(`--output-uri` to override).

## Tests

```bash
pytest
```
