"""Dataset cleaning, incremental upsert and embedding of only new/changed rows."""

import hashlib
import html
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from sqlalchemy import bindparam, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models import Job
from app.services.embeddings import DOC, Embedder
from app.services.skills import extract_skills

log = logging.getLogger("app.ingest")

DATA_URL = "https://huggingface.co/datasets/datastax/linkedin_job_listings/resolve/main/postings.csv"
RAW_PATH = Path("data/raw/postings_head.csv")
HEAD_BYTES = 60_000_000  # first ~60MB (~14.6k postings, 2 weeks of Apr 2024) of the 517MB CSV
MIN_DESC_CHARS = 200
EMBED_CHARS = 1500  # truncate embedding text: keeps free-tier token use low


@dataclass
class CleanJob:
    source_id: str
    title: str
    company: str | None
    location: str | None
    remote: bool
    description: str
    skills: list[str]
    experience_level: str | None
    posted_at: datetime | None
    content_hash: str
    url: str | None = None
    apply_url: str | None = None
    source: str = "linkedin_2024"
    expires_at: datetime | None = None
    last_seen_at: datetime | None = None


_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"[ \t\r\f\v]+")
_NL = re.compile(r"\n{3,}")


def clean_text(s: str) -> str:
    s = html.unescape(_TAG.sub(" ", s or ""))
    s = _WS.sub(" ", s)
    return _NL.sub("\n\n", s).strip()


def content_hash(title: str, company: str | None, location: str | None, desc: str) -> str:
    return hashlib.sha256(f"{title}|{company}|{location}|{desc}".encode()).hexdigest()


_SCHEME = re.compile(r"^[a-z][a-z0-9+.\-]*:", re.I)
_BARE_DOMAIN = re.compile(r"^(www\.)?[\w\-]+(\.[\w\-]+)+(/|$)")


def normalize_link(u: str | None, drop_params: tuple[str, ...] = ()) -> str | None:
    """Return a safe http(s) URL or None. Rejects javascript:, data:, mailto:, etc. outright.

    Dataset links are untrusted input that ends up in an <a href>, so only http and https survive.
    Bare domains ("www.example.org") get https. `drop_params` removes tracking parameters.
    """
    u = (u or "").strip()
    if not u or u.lower() in ("nan", "none", "null") or len(u) > 1000:
        return None
    if not _SCHEME.match(u):
        if not _BARE_DOMAIN.match(u):
            return None
        u = "https://" + u
    parts = urlsplit(u)
    if parts.scheme.lower() not in ("http", "https") or not parts.netloc or " " in parts.netloc:
        return None
    if drop_params and parts.query:
        q = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if k not in drop_params]
        parts = parts._replace(query=urlencode(q))
    return urlunsplit(parts)


def _ms_to_dt(v) -> datetime | None:
    try:
        return datetime.fromtimestamp(float(v) / 1000, tz=UTC) if v == v and v else None
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def _val(row: dict, key: str) -> str:
    """Cell as text; pandas NaN / None / the string 'nan' become ''."""
    v = row.get(key)
    if v is None or (isinstance(v, float) and v != v):
        return ""
    s = str(v)
    return "" if s.strip().lower() in ("nan", "none", "null") else s


def clean_row(row: dict) -> CleanJob | None:
    title = clean_text(_val(row, "title"))
    desc = clean_text(_val(row, "description"))
    if not title or len(desc) < MIN_DESC_CHARS:
        return None
    company = clean_text(_val(row, "company_name")) or None
    location = clean_text(_val(row, "location")) or None
    remote_flag = str(row.get("remote_allowed") or "").strip() in ("1", "1.0", "True", "true")
    remote = remote_flag or bool(re.search(r"\bremote\b", f"{title} {location or ''}", re.I))
    ts = row.get("listed_time")
    try:
        posted = datetime.fromtimestamp(float(ts) / 1000, tz=UTC) if ts == ts and ts else None
    except (TypeError, ValueError):
        posted = None
    level = clean_text(_val(row, "formatted_experience_level")) or None
    skills = extract_skills(f"{title}\n{desc}\n{_val(row, 'skills_desc')}")
    return CleanJob(
        source_id=str(row["job_id"]).split(".")[0], title=title[:500], company=company,
        location=location, remote=remote, description=desc, skills=skills,
        experience_level=level, posted_at=posted,
        content_hash=content_hash(title, company, location, desc),
        url=normalize_link(_val(row, "job_posting_url"), drop_params=("trk", "trackingId", "refId")),
        apply_url=normalize_link(_val(row, "application_url")),
        source="linkedin_2024", expires_at=_ms_to_dt(row.get("expiry")),
    )


def load_feed(path: Path = RAW_PATH) -> list[CleanJob]:
    """Read + clean + dedupe the CSV, in a fixed pseudo-random arrival order (a simulated feed)."""
    import pandas as pd

    df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    df = df.dropna(subset=["job_id", "title", "description"])
    df["job_id"] = df["job_id"].astype("int64")
    df = df.drop_duplicates("job_id")
    # Simulated arrival order: a deterministic shuffle, so the first N postings span the whole
    # two-week window (time order would put the first 5,000 inside a single 5-hour slice).
    df["_arrival"] = (df["job_id"] * 2654435761) % (2**32)
    df = df.sort_values(["_arrival", "job_id"])
    jobs, seen = [], set()
    for row in df.to_dict("records"):
        j = clean_row(row)
        if j is None:
            continue
        sig = (j.title.lower(), (j.company or "").lower(), (j.location or "").lower())
        if sig in seen:
            continue
        seen.add(sig)
        jobs.append(j)
    return jobs


def embedding_text(j: Job | CleanJob) -> str:
    parts = [j.title, j.company or "", j.location or "", ", ".join(j.skills or []),
             (j.description or "")[:EMBED_CHARS]]
    return "\n".join(p for p in parts if p)


@dataclass
class IngestStats:
    inserted: int = 0
    updated: int = 0
    unchanged: int = 0
    embedded: int = 0
    meta_updated: int = 0


async def upsert_jobs(session: AsyncSession, feed: list[CleanJob], add_new: int) -> IngestStats:
    """Process the feed prefix covering existing rows + `add_new` new ones.

    Unchanged rows are skipped; changed rows get embedding=NULL so they are re-embedded;
    new rows are inserted without an embedding.
    """
    stats = IngestStats()
    existing = {
        r[0]: (r[1], r[2], r[3], r[4], r[5])
        for r in (await session.execute(select(
            Job.source_id, Job.content_hash, Job.url, Job.apply_url, Job.expires_at, Job.last_seen_at
        ))).all()
    }
    new_budget = add_new
    meta_batch: list[dict] = []  # same text, new link/expiry/freshness: one batched UPDATE
    for j in feed:
        if j.source_id in existing:
            old_hash, *old_meta = existing[j.source_id]
            if old_hash == j.content_hash:
                # same text: no re-embedding. Only refresh link / expiry / freshness metadata.
                if tuple(old_meta) != (j.url, j.apply_url, j.expires_at, j.last_seen_at):
                    meta_batch.append({"b_sid": j.source_id, "b_url": j.url, "b_apply": j.apply_url,
                                       "b_exp": j.expires_at, "b_seen": j.last_seen_at})
                    stats.meta_updated += 1
                else:
                    stats.unchanged += 1
                continue
            await session.execute(
                update(Job).where(Job.source_id == j.source_id).values(
                    title=j.title, company=j.company, location=j.location, remote=j.remote,
                    description=j.description, skills=j.skills, experience_level=j.experience_level,
                    posted_at=j.posted_at, content_hash=j.content_hash, embedding=None,
                    embedding_model=None, updated_at=datetime.now(UTC), url=j.url,
                    apply_url=j.apply_url, expires_at=j.expires_at, last_seen_at=j.last_seen_at,
                )
            )
            stats.updated += 1
        elif new_budget > 0:
            session.add(Job(**j.__dict__))
            new_budget -= 1
            stats.inserted += 1
    if meta_batch:
        table = Job.__table__
        await session.execute(
            update(table).where(table.c.source_id == bindparam("b_sid")).values(
                url=bindparam("b_url"), apply_url=bindparam("b_apply"),
                expires_at=bindparam("b_exp"), last_seen_at=bindparam("b_seen"),
            ),
            meta_batch,
        )
    await session.commit()
    return stats


async def embed_pending(session: AsyncSession, embedder: Embedder, chunk: int = 64) -> int:
    """Embed rows whose embedding is NULL. Commits per chunk, so it is resumable."""
    model = get_settings().embed_model
    total = 0
    while True:
        rows = (
            await session.execute(
                select(Job).where(Job.embedding.is_(None)).order_by(Job.id).limit(chunk)
            )
        ).scalars().all()
        if not rows:
            return total
        vecs = await embedder.embed([embedding_text(r) for r in rows], DOC)
        for r, v in zip(rows, vecs, strict=True):
            r.embedding, r.embedding_model = v, model
        await session.commit()
        total += len(rows)
        log.info("embedded", extra={"batch": len(rows), "total": total})


async def analyze(session: AsyncSession) -> None:
    await session.execute(text("ANALYZE jobs"))
    await session.commit()
