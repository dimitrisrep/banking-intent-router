# Banking Intent Router

Classical NLP classification of banking-support requests into 77 intents, with held-out calibration and a human-review threshold.

The project demonstrates a complete text-classification experiment: dataset verification, duplicate isolation, separate fitting/calibration/selection partitions, model comparison, untouched-test reporting and API inference. It uses the public BANKING77 dataset rather than generated text for the reported benchmark.

## Measured result

The selected word/character TF-IDF + SGD logistic model achieved the following on the original 3,080-row test set:

| Metric | Value |
| --- | ---: |
| Accuracy | 0.8919 |
| Macro F1 | 0.8921 |
| Top-3 accuracy | 0.9653 |
| Accepted-request coverage | 0.9958 |
| Accepted-request accuracy | 0.8950 |
| Expected calibration error, 15 bins | 0.1317 |

The validation-selected threshold was 0.19. It was chosen to maximize validation coverage with at least 90% accepted accuracy and 100 accepted requests. The test result fell below that accuracy target. The model is not a validated out-of-domain detector and must not be described as a production-ready routing system.

See [the complete report](reports/metrics.json) for per-class metrics, the 77-class confusion matrix, calibration metrics, validation risk/coverage curve and local latency measurements. [errors.json](reports/errors.json) records error row indices without duplicating the raw dataset.

## Reproduce

Use Python 3.12 for the locked environment. Run commands from this project directory.

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.lock
pip install --no-deps -e .

intent-router download
intent-router train
intent-router predict "My card has been stolen. What should I do?"
uvicorn intent_router.api:app --host 127.0.0.1 --port 8001
```

The downloader pins the upstream commit and checks both SHA-256 values. Benchmark data and model artifacts are not committed. A subsequent run verifies cached files before training. See [DATA_CARD.md](docs/DATA_CARD.md) for the source, citation and license.

## Experiment design

1. Load the official train/test CSVs after checksum verification.
2. Normalize only text keys for duplicate detection. Do not rewrite model inputs.
3. Remove four duplicate training rows and seven normalized training texts that also occur in test. Test labels are not used for this cleaning step.
4. Partition the remaining 9,992 training examples into 5,995 fit, 1,998 calibration and 1,999 validation examples, stratified by intent.
5. Fit word/character TF-IDF with a linear SVM and with an SGD logistic classifier on the fit partition only.
6. Freeze each entire fitted pipeline and fit sigmoid calibration on the independent calibration partition.
7. Select the model and review threshold using validation only.
8. Report results once on the unchanged official test set. Also report deduplicated-test results because the official test has repeated text.

The vocabulary, classifier weights and calibration do not see validation or test during fitting. Selection and threshold tuning do not inspect test labels. The serving model is the same fitted/calibrated model evaluated on test; no full-dataset refit is hidden behind the reported metrics.

## API

```bash
curl http://127.0.0.1:8001/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"My card has been stolen","top_k":3}'
```

Responses include `routed` or `human_review`, ranked candidate intents, calibrated scores and the selected threshold. `GET /health` reports model state, and `GET /ready` returns 503 if no artifact is available. Input is bounded to 4,000 characters; no messages are persisted.

Set `INTENT_ROUTER_MODEL` to the local `router.joblib` path. Joblib files are executable serialized data: load only your own trusted training output. The loader requires the training scikit-learn version.

## Tests and container

```bash
pytest
ruff check src tests
docker build -t banking-intent-router .
docker run --rm -p 127.0.0.1:8001:8001 banking-intent-router
```

Train before the Docker build so `artifacts/router.joblib` exists. The image uses the locked dependencies and a non-root user. The unit tests use small authored fixtures and do not require a network connection. They cover checksums, split isolation, threshold logic, real sparse-model inference, API validation and unavailable-model behavior. Remote CI and Docker execution must be verified independently of local Python tests.

## Scope and next experiments

English banking requests only. There is no production traffic evaluation, transformer benchmark, validated OOD detection, fairness assessment or drift monitor. Calibration error remains material. Future comparisons should use a new development protocol and preserve an unused final evaluation set; repeatedly tuning against this test report would invalidate its role as an unbiased check.

Potential extensions include a frozen-encoder baseline, explicit out-of-scope data, label-error review and a drift/evaluation dashboard. These are future experiments, not implemented capabilities.
