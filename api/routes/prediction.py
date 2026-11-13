from fastapi import APIRouter, HTTPException

from api.schemas import (ClaimTriageResponse, FrequencyResponse, PolicyInput,
                         SegmentResponse, SeverityResponse)
from api.services import models

router = APIRouter(prefix="/predict", tags=["Prediction"])


def _run(fn, policy: PolicyInput):
    try:
        return fn(policy.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Model file missing: {exc.filename}")


@router.post("/frequency", response_model=FrequencyResponse)
def frequency(policy: PolicyInput):
    return _run(models.predict_frequency, policy)


@router.post("/severity", response_model=SeverityResponse)
def severity(policy: PolicyInput):
    return _run(models.predict_severity, policy)


@router.post("/segment", response_model=SegmentResponse)
def segment(policy: PolicyInput):
    return _run(models.predict_segment, policy)


@router.post("/triage", response_model=ClaimTriageResponse)
def triage(policy: PolicyInput):
    """One call that combines all three models - this is what n8n uses."""
    data = policy.model_dump()
    try:
        freq = models.predict_frequency(data)
        sev = models.predict_severity(data)
        seg = models.predict_segment(data)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Model file missing: {exc.filename}")
    return {
        "expected_claims_per_year": freq["expected_claims_per_year"],
        "risk_level": freq["risk_level"],
        "predicted_claim_amount": sev["predicted_claim_amount"],
        "segment_name": seg["segment_name"],
        "recommended_action": models.recommended_action(
            freq["risk_level"], sev["predicted_claim_amount"]),
    }
