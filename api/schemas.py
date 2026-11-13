"""Request / response models for the MotorAI API."""

from typing import Optional

from pydantic import BaseModel, Field


class PolicyInput(BaseModel):
    """Policy / driver details used by every prediction endpoint."""

    Exposure: float = Field(0.75, gt=0, le=1.5, description="Fraction of a year insured")
    VehPower: int = Field(7, ge=1, le=20)
    VehAge: int = Field(3, ge=0, le=100)
    DrivAge: int = Field(35, ge=18, le=100)
    BonusMalus: int = Field(60, ge=50, le=230)
    VehBrand: str = Field("B12")
    VehGas: str = Field("Regular", description="Regular or Diesel")
    Area: str = Field("C", description="Area code A-F")
    Density: int = Field(500, ge=1, description="Inhabitants per km2")
    Region: str = Field("Ile-de-France")


class FrequencyResponse(BaseModel):
    expected_claims_per_year: float
    expected_claims_for_period: float
    risk_level: str
    model: str


class SeverityResponse(BaseModel):
    predicted_claim_amount: float
    model: str


class SegmentResponse(BaseModel):
    cluster: int
    segment_name: str
    description: str


class ClaimTriageResponse(BaseModel):
    """Combined answer used by the n8n workflow."""

    expected_claims_per_year: float
    risk_level: str
    predicted_claim_amount: float
    segment_name: str
    recommended_action: str


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=3)


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    mode: str
    llm_error: Optional[str] = None
