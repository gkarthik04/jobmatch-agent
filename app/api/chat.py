from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.deps import get_embedder, get_llm_dep
from app.models import Message, User
from app.ratelimit import rate_limit
from app.security import current_user
from app.services.agent import run_agent
from app.services.agent_tools import AgentContext

router = APIRouter(tags=["chat"])


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


@router.post("/chat", dependencies=[Depends(rate_limit)])
async def chat(
    body: ChatIn,
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
    embedder=Depends(get_embedder),
    llm=Depends(get_llm_dep),
):
    """Run the tool-calling agent for one user turn. Returns the reply plus every step it took."""
    ctx = AgentContext(session=session, user_id=user.id, embedder=embedder, llm=llm)
    result = await run_agent(ctx, body.message)
    return result.to_dict()


@router.get("/chat/history")
async def history(user: User = Depends(current_user), session: AsyncSession = Depends(get_session)):
    rows = (
        await session.execute(
            select(Message).where(Message.user_id == user.id).order_by(Message.id.desc()).limit(40)
        )
    ).scalars().all()[::-1]
    return [{"role": m.role, "content": m.content} for m in rows]


@router.delete("/chat/history", status_code=204)
async def reset(user: User = Depends(current_user), session: AsyncSession = Depends(get_session)):
    await session.execute(delete(Message).where(Message.user_id == user.id))
    await session.commit()
