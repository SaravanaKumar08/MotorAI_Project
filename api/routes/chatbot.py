from fastapi import APIRouter

from api.schemas import ChatRequest, ChatResponse
from api.services import policy_assistant

router = APIRouter(prefix="/chat", tags=["Policy Assistant"])


@router.post("", response_model=ChatResponse)
def chat(request: ChatRequest):
    return policy_assistant.answer(request.question)
