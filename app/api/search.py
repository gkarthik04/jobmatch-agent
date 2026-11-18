from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.deps import get_embedder, get_llm_dep
from app.schemas import SearchResponse, job_out
from app.services.search import Mode, SearchFilters, dedupe_hits, search

router = APIRouter(tags=["search"])


@router.get("/search", response_model=SearchResponse)
async def search_jobs(
    q: str = Query(..., min_length=2, max_length=500),
    mode: Mode = "hybrid",
    limit: int = Query(10, ge=1, le=50),
    location: str | None = None,
    remote: bool | None = None,
    posted_within_days: int | None = Query(None, ge=1, le=365),
    session: AsyncSession = Depends(get_session),
    embedder=Depends(get_embedder),
    llm=Depends(get_llm_dep),
):
    """Hybrid job search. `mode`: vector | fts | hybrid | hybrid_rerank."""
    filters = SearchFilters(location=location, remote=remote, posted_within_days=posted_within_days)
    hits = await search(session, embedder, q, filters=filters, mode=mode, limit=limit * 3, llm=llm)
    hits = dedupe_hits(hits, limit=limit)  # one posting listed in many cities appears once
    return SearchResponse(query=q, mode=mode, count=len(hits), results=[job_out(h) for h in hits])
