"""Sparse text models, held-out sigmoid calibration and review thresholds."""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from intent_router.data import CHECKSUMS, REVISION, load, partition, text_keys


def make_pipeline(name="linear_svm"):
    estimator = (
        LinearSVC(C=1, random_state=42)
        if name == "linear_svm"
        else SGDClassifier(
            loss="log_loss", alpha=1e-5, max_iter=1000, tol=1e-4, random_state=42
        )
    )
    return Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "word",
                            TfidfVectorizer(
                                ngram_range=(1, 2),
                                sublinear_tf=True,
                                min_df=2,
                                max_features=40000,
                            ),
                        ),
                        (
                            "character",
                            TfidfVectorizer(
                                analyzer="char_wb",
                                ngram_range=(3, 5),
                                min_df=2,
                                max_features=40000,
                            ),
                        ),
                    ]
                ),
            ),
            ("classifier", estimator),
        ]
    )


def fit_calibrated(name, fit, calibration):
    base = make_pipeline(name).fit(fit.text.tolist(), fit.category.tolist())
    # FrozenEstimator guarantees the feature vocabulary and base model stay fixed.
    calibrated = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid").fit(
        calibration.text.tolist(), calibration.category.tolist()
    )
    return base, calibrated


def select_threshold(labels, predicted, scores, target_accuracy=0.9, min_count=100):
    choices = []
    for threshold in np.linspace(0, 1, 101):
        mask = scores >= threshold
        accuracy = (
            float(np.mean(np.asarray(labels)[mask] == np.asarray(predicted)[mask]))
            if mask.any()
            else None
        )
        choices.append(
            {
                "threshold": float(threshold),
                "coverage": float(mask.mean()),
                "accepted_count": int(mask.sum()),
                "accepted_accuracy": accuracy,
            }
        )
    eligible = [
        r
        for r in choices
        if r["accepted_count"] >= min_count
        and r["accepted_accuracy"] >= target_accuracy
    ]
    threshold = (
        max(eligible, key=lambda r: r["coverage"])["threshold"] if eligible else 1.0
    )
    return threshold, choices


def expected_calibration_error(labels, predicted, scores):
    correct = np.asarray(labels) == np.asarray(predicted)
    error = 0.0
    for left, right in zip(np.linspace(0, 1, 16)[:-1], np.linspace(0, 1, 16)[1:]):
        mask = (scores >= left) & ((scores < right) if right < 1 else (scores <= right))
        if mask.any():
            error += mask.mean() * abs(correct[mask].mean() - scores[mask].mean())
    return float(error)


def score_report(labels, probabilities, classes, threshold):
    labels = np.asarray(labels)
    predicted = classes[np.argmax(probabilities, axis=1)]
    scores = probabilities.max(axis=1)
    accepted = scores >= threshold
    top3 = classes[np.argsort(probabilities, axis=1)[:, -3:]]
    return {
        "accuracy": accuracy_score(labels, predicted),
        "macro_f1": f1_score(labels, predicted, average="macro", zero_division=0),
        "top3_accuracy": float(np.mean([y in row for y, row in zip(labels, top3)])),
        "log_loss": log_loss(labels, probabilities, labels=classes),
        "ece_15_bins": expected_calibration_error(labels, predicted, scores),
        "coverage": float(accepted.mean()),
        "accepted_count": int(accepted.sum()),
        "accepted_accuracy": float(np.mean(labels[accepted] == predicted[accepted]))
        if accepted.any()
        else None,
        "classification_report": classification_report(
            labels, predicted, output_dict=True, zero_division=0
        ),
        "classes": classes.tolist(),
        "confusion_matrix": confusion_matrix(
            labels, predicted, labels=classes
        ).tolist(),
    }


def train(directory: Path, output: Path):
    training, test = load(directory)
    splits, audit = partition(training, test)
    fit, calibration, validation = (
        splits[key] for key in ["fit", "calibration", "validation"]
    )
    candidates = {}
    selected = None
    for name in ["linear_svm", "sgd_logistic"]:
        base, model = fit_calibrated(name, fit, calibration)
        predicted = model.predict(validation.text.tolist())
        score = float(f1_score(validation.category, predicted, average="macro"))
        candidates[name] = {"validation_macro_f1": score}
        if selected is None or score > selected[0]:
            selected = (score, name, base, model)
    _, name, base, model = selected
    validation_prob = model.predict_proba(validation.text.tolist())
    threshold, curve = select_threshold(
        validation.category,
        model.classes_[validation_prob.argmax(axis=1)],
        validation_prob.max(axis=1),
    )
    # Selection and thresholding have finished. Read test labels only for reporting.
    probabilities = model.predict_proba(test.text.tolist())
    baseline = DummyClassifier(strategy="most_frequent").fit(
        np.zeros((len(fit), 1)), fit.category
    )
    report = {
        "dataset": "BANKING77",
        "upstream_revision": REVISION,
        "checksums": CHECKSUMS,
        "seed": 42,
        "sklearn_version": sklearn.__version__,
        "audit": audit,
        "partition_sizes": {key: len(value) for key, value in splits.items()},
        "split_protocol": "60% fit / 20% calibration / 20% validation from cleaned official train; official test evaluated once",
        "calibration": "held-out sigmoid, frozen complete base pipeline",
        "models": candidates,
        "selected_model": name,
        "threshold": threshold,
        "threshold_target": "maximize validation coverage at >=90% accepted accuracy and >=100 accepted examples",
        "validation_risk_coverage": curve,
        "test": score_report(test.category, probabilities, model.classes_, threshold),
        "majority_test_macro_f1": f1_score(
            test.category, baseline.predict(np.zeros((len(test), 1))), average="macro"
        ),
        "limitations": [
            "English banking requests only; no validated out-of-domain detector.",
            "Calibration and routing targets do not guarantee deployment precision.",
            "Sparse classical baselines; no transformer performance is claimed.",
        ],
    }
    unique = ~text_keys(test.text).duplicated()
    report["deduplicated_test"] = {
        "rows": int(unique.sum()),
        "accuracy": accuracy_score(
            test.category[unique], model.classes_[probabilities[unique].argmax(axis=1)]
        ),
        "macro_f1": f1_score(
            test.category[unique],
            model.classes_[probabilities[unique].argmax(axis=1)],
            average="macro",
        ),
    }
    timings = []
    for text in test.text.iloc[:25]:
        start = time.perf_counter()
        model.predict_proba([text])
        timings.append((time.perf_counter() - start) * 1000)
    report["single_request_latency_ms"] = {
        "sample_size": 25,
        "median": float(np.median(timings)),
        "p95": float(np.quantile(timings, 0.95)),
        "scope": "local warm CPU; not a load test",
    }
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "schema_version": 1,
            "sklearn_version": sklearn.__version__,
            "model": model,
            "base": base,
            "threshold": threshold,
            "dataset_revision": REVISION,
        },
        output / "router.joblib",
    )
    (output / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    predictions = model.classes_[probabilities.argmax(axis=1)]
    errors = [
        {
            "test_row": int(i),
            "true_label": str(test.category.iloc[i]),
            "predicted_label": str(label),
            "score": float(probabilities[i].max()),
        }
        for i, label in enumerate(predictions)
        if label != test.category.iloc[i]
    ]
    (output / "errors.json").write_text(
        json.dumps(errors, indent=2) + "\n", encoding="utf-8"
    )
    return report


class IntentRouter:
    def __init__(self, bundle):
        self.bundle = bundle

    @classmethod
    def load(cls, path: Path):
        bundle = joblib.load(path)  # Trusted local artifacts only.
        if (
            bundle.get("schema_version") != 1
            or bundle.get("sklearn_version") != sklearn.__version__
        ):
            raise ValueError("Artifact schema or sklearn version mismatch; retrain.")
        return cls(bundle)

    def predict(self, text: str, top_k: int = 3):
        if not text.strip() or len(text) > 4000 or not 1 <= top_k <= 10:
            raise ValueError("Text must contain 1-4000 characters; top_k must be 1-10.")
        model = self.bundle["model"]
        probabilities = model.predict_proba([text])[0]
        order = np.argsort(probabilities)[::-1][:top_k]
        results = [
            {"intent": str(model.classes_[i]), "score": float(probabilities[i])}
            for i in order
        ]
        accepted = results[0]["score"] >= self.bundle["threshold"]
        return {
            "status": "routed" if accepted else "human_review",
            "candidates": results,
            "threshold": self.bundle["threshold"],
            "scope": "English banking intents",
            "score_type": "heldout_sigmoid_calibration",
        }
