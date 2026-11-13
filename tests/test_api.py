from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

POLICY = {
    "Exposure": 0.75, "VehPower": 7, "VehAge": 3, "DrivAge": 35,
    "BonusMalus": 60, "VehBrand": "B12", "VehGas": "Regular",
    "Area": "C", "Density": 500, "Region": "Ile-de-France",
}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_frequency():
    body = client.post("/predict/frequency", json=POLICY).json()
    assert body["expected_claims_per_year"] >= 0
    assert body["risk_level"] in {"LOW", "MEDIUM", "HIGH"}


def test_severity():
    body = client.post("/predict/severity", json=POLICY).json()
    assert body["predicted_claim_amount"] >= 0


def test_segment():
    body = client.post("/predict/segment", json=POLICY).json()
    assert body["cluster"] in {0, 1, 2}


def test_triage_has_action():
    body = client.post("/predict/triage", json=POLICY).json()
    assert body["recommended_action"] in {
        "FAST_TRACK", "STANDARD_PROCESSING", "MANUAL_REVIEW"}


def test_invalid_input_rejected():
    bad = dict(POLICY, DrivAge=5)
    assert client.post("/predict/triage", json=bad).status_code == 422


def test_chat_returns_sources():
    body = client.post("/chat", json={
        "question": "What documents are required for a motor insurance claim?"}).json()
    assert body["sources"]
    assert body["answer"]
