"""MotorAI REST API.   Run locally:  uvicorn api.main:app --reload"""

from fastapi import FastAPI

from api.routes import chatbot, prediction

app = FastAPI(
    title="MotorAI - Intelligent Motor Insurance API",
    description="Claim frequency, claim severity, customer segmentation and a policy assistant.",
    version="1.0.0",
)

app.include_router(prediction.router)
app.include_router(chatbot.router)


@app.get("/", tags=["Health"])
def root():
    return {"project": "MotorAI", "status": "running", "docs": "/docs"}


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}
