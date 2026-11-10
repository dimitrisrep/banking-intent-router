import hashlib

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from intent_router.api import create_app
from intent_router.data import partition, verify
from intent_router.model import (
    IntentRouter,
    make_pipeline,
    score_report,
    select_threshold,
)


@pytest.fixture(scope="module")
def router():
    texts = [
        "my card was stolen",
        "I lost the debit card",
        "someone took my card",
        "how can I change the pin",
        "change my secret pin number",
        "set a new pin",
        "when will my transfer arrive",
        "bank transfer waiting time",
        "my transfer is pending",
    ]
    labels = ["lost_card"] * 3 + ["change_pin"] * 3 + ["transfer_timing"] * 3
    base = make_pipeline().fit(texts, labels)
    calibration_texts = [
        "report a stolen debit card",
        "the card is missing",
        "my wallet and card were stolen",
        "lost the physical card",
        "someone stole my payment card",
        "reset the pin code",
        "where can I change a pin",
        "choose a new pin number",
        "my old pin needs changing",
        "update the card pin",
        "how long does a bank transfer take",
        "expected transfer arrival time",
        "waiting for the money transfer",
        "transfer not arrived yet",
        "duration of a bank transfer",
    ]
    calibrated = CalibratedClassifierCV(FrozenEstimator(base)).fit(
        calibration_texts,
        ["lost_card"] * 5 + ["change_pin"] * 5 + ["transfer_timing"] * 5,
    )
    return IntentRouter({"model": calibrated, "threshold": 0.5, "base": base})


def test_checksum_guard(tmp_path):
    path = tmp_path / "file.csv"
    path.write_bytes(b"safe")
    verify(path, hashlib.sha256(b"safe").hexdigest())
    with pytest.raises(ValueError):
        verify(path, "0" * 64)


def test_partition_has_no_duplicate_or_test_text_leakage():
    frame = pd.DataFrame(
        [
            {"text": f"class {label} request {i}", "category": label}
            for label in ["A", "B", "C"]
            for i in range(30)
        ]
    )
    test = frame.iloc[[0, 30, 60]].copy()
    frame = pd.concat([frame, frame.iloc[[1]]])
    splits, audit = partition(frame, test)
    assert audit["duplicate_train_rows_removed"] == 1
    assert audit["test_overlap_train_rows_removed"] == 3
    all_keys = [set(splits[k].key) for k in splits]
    assert not all_keys[0] & all_keys[1]
    assert not all_keys[0] & all_keys[2]
    assert not set(test.text) & set.union(*all_keys)


def test_threshold_selection_and_unattainable_target():
    threshold, _ = select_threshold(
        ["A", "A", "B"], ["A", "B", "B"], np.array([0.9, 0.3, 0.95]), min_count=2
    )
    assert 0.3 < threshold <= 0.9
    threshold, _ = select_threshold(["A"], ["B"], np.array([0.9]), min_count=1)
    assert threshold == 1.0


def test_real_sparse_model_predictions(router):
    response = router.predict("my card was stolen")
    assert response["candidates"][0]["intent"] == "lost_card"
    assert len(response["candidates"]) == 3
    assert 0 <= response["candidates"][0]["score"] <= 1


@pytest.mark.parametrize("text", ["", " ", "a" * 4001])
def test_reject_invalid_input(router, text):
    with pytest.raises(ValueError):
        router.predict(text)


def test_metrics_include_abstention():
    result = score_report(
        ["A", "B"], np.array([[0.9, 0.1], [0.6, 0.4]]), np.array(["A", "B"]), 0.8
    )
    assert result["coverage"] == 0.5
    assert result["accepted_accuracy"] == 1
    assert result["accuracy"] == 0.5


def test_api_validation_unavailable_and_inference(router):
    with TestClient(create_app(router)) as client:
        assert client.get("/ready").status_code == 200
        assert client.post("/predict", json={"text": "lost card"}).status_code == 200
        assert client.post("/predict", json={"text": " "}).status_code == 422
        assert (
            client.post("/predict", json={"text": "lost card", "top_k": 11}).status_code
            == 422
        )


def test_missing_artifact_is_explicit(monkeypatch, tmp_path):
    monkeypatch.setenv("INTENT_ROUTER_MODEL", str(tmp_path / "missing.joblib"))
    with TestClient(create_app()) as client:
        assert client.get("/health").json()["model_ready"] is False
        assert client.get("/ready").status_code == 503
        assert client.post("/predict", json={"text": "lost card"}).status_code == 503
