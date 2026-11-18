"""Live job feed: currently open positions with real apply links.

Source: the Arbeitnow public job board API (free, no key, documented for external use; mostly
Europe). Every job keeps its Arbeitnow URL as the apply link and is shown with a "via Arbeitnow"
credit. We fetch a few pages once a day; `last_seen_at` marks a job as still open, and a job that
stops appearing in the feed is treated as closed after OPEN_WINDOW_DAYS.
"""

import hashlib
import html
import json
import logging
import re
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime

from app.services.ingest import MIN_DESC_CHARS, CleanJob, clean_text, content_hash, normalize_link
from app.services.skills import extract_skills

log = logging.getLogger("app.live_feed")

ARBEITNOW_URL = "https://www.arbeitnow.com/api/job-board-api"
USER_AGENT = "jobmatch-portfolio/1.0 (open source demo; contact via GitHub)"
SOURCE = "arbeitnow"
OPEN_WINDOW_DAYS = 3

_EN = {"the", "and", "you", "we", "with", "for", "our", "will", "your", "to", "of", "in", "is", "are",
       "this", "that", "have", "experience", "team", "work", "as", "be", "on", "an"}
_OTHER = {  # stopwords of languages the English embedding model cannot match well
    "de": {"und", "der", "die", "das", "wir", "für", "mit", "sie", "ein", "eine", "ist", "den", "von", "zu",
           "auf", "bei", "als", "nicht", "oder", "werden", "auch"},
    "nl": {"het", "een", "van", "voor", "met", "je", "jij", "wij", "bij", "aan", "door", "naar", "ook",
           "heel", "waar", "binnen", "onze", "ons", "gaat", "werken"},
    "fr": {"le", "la", "les", "des", "pour", "vous", "nous", "avec", "une", "est", "dans", "sur", "que",
           "qui", "notre", "votre", "ce", "au", "du"},
    "es": {"el", "los", "las", "para", "con", "una", "por", "que", "nuestro", "como", "del", "se"},
    "it": {"il", "gli", "per", "con", "una", "che", "nostro", "della", "sono", "del", "alle"},
    "pt": {"os", "as", "para", "com", "uma", "que", "nosso", "dos", "das", "não"},
    "sv": {"och", "att", "det", "som", "för", "med", "vi", "är", "på", "en", "av"},
}


_PROMO = re.compile(r"\s*Find (more )?(English Speaking )?Jobs in [A-Za-z ]+ on Arbeitnow\s*$", re.I)


def make_source_id(slug: str) -> str:
    """Stable short id: slugs can exceed the 64 character source_id column."""
    return f"{SOURCE}:{hashlib.sha1(slug.encode()).hexdigest()[:24]}"


_BLOCK_END = re.compile(r"(?i)<\s*(br\s*/?|/p|/div|/li|/ul|/ol|/h[1-6]|/tr)\s*>")
_LI_START = re.compile(r"(?i)<\s*li(\s[^>]*)?>")


def _tidy(s: str) -> str:
    """Feed HTML -> readable text.

    The feed double escapes its HTML (&lt;p&gt;...), so we unescape first, turn block ends into
    line breaks, strip the tags, and unescape again. Also drops a promo footer and U+FFFD.
    """
    s = html.unescape(s or "")
    s = _LI_START.sub("\n• ", s)
    s = _BLOCK_END.sub("\n", s)
    s = clean_text(s).replace("\ufffd", "-")
    s = re.sub(r"[ \t]*\n[ \t]*", "\n", s)
    s = re.sub(r"\n{2,}(?=• )", "\n", s)  # keep list items together
    return _PROMO.sub("", s).strip()


def looks_english(text_: str) -> bool:
    """Cheap stopword vote. The index uses an English embedding model, so we skip German postings."""
    words = re.findall(r"[a-zà-öø-ÿß]+", text_.lower())[:400]
    en = sum(w in _EN for w in words)
    other = max(sum(w in stop for w in words) for stop in _OTHER.values())
    return en >= 5 and en >= 2 * other


def fetch_pages(pages: int = 3, timeout: float = 30.0, delay_s: float = 1.0) -> list[dict]:
    """GET up to `pages` pages from the feed (blocking; call from a thread)."""
    jobs: list[dict] = []
    for page in range(1, pages + 1):
        req = urllib.request.Request(f"{ARBEITNOW_URL}?page={page}", headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - fixed https URL
                data = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and jobs:  # rate limited: keep what we have, try again tomorrow
                log.warning("live feed rate limited", extra={"page": page})
                break
            raise
        jobs.extend(data.get("data", []))
        if not data.get("links", {}).get("next"):
            break
        time.sleep(delay_s)
    return jobs


def clean_arbeitnow(raw: dict, now: datetime | None = None) -> CleanJob | None:
    now = now or datetime.now(UTC)
    title = _tidy(str(raw.get("title") or ""))
    desc = _tidy(str(raw.get("description") or ""))
    slug = str(raw.get("slug") or "").strip()
    url = normalize_link(raw.get("url"))
    if not (title and slug and url) or len(desc) < MIN_DESC_CHARS:
        return None
    if not looks_english(f"{title} {desc}"):
        return None
    company = clean_text(str(raw.get("company_name") or "")) or None
    location = clean_text(str(raw.get("location") or "")) or None
    tags = [str(t) for t in (raw.get("tags") or [])]
    try:
        posted = datetime.fromtimestamp(float(raw["created_at"]), tz=UTC)
    except (KeyError, TypeError, ValueError, OverflowError, OSError):
        posted = None
    return CleanJob(
        source_id=make_source_id(slug), title=title[:500], company=company, location=location,
        remote=bool(raw.get("remote")), description=desc,
        skills=extract_skills(f"{title}\n{desc}\n{' '.join(tags)}"),
        experience_level=None, posted_at=posted,
        content_hash=content_hash(title, company, location, desc),
        url=url, apply_url=url, source=SOURCE, expires_at=None, last_seen_at=now,
    )


def clean_feed(raw_jobs: list[dict], now: datetime | None = None) -> list[CleanJob]:
    seen: set[str] = set()
    out: list[CleanJob] = []
    for raw in raw_jobs:
        j = clean_arbeitnow(raw, now)
        if j and j.source_id not in seen:
            seen.add(j.source_id)
            out.append(j)
    return out


async def repair_existing(session) -> dict[str, int]:
    """Clean up stored live rows: re-render descriptions and close non-English postings.

    Safe to run every time. Rewritten descriptions get a new content hash and a NULL embedding,
    so only those rows are re-embedded.
    """
    from sqlalchemy import select, update

    from app.models import Job

    fixed = closed = 0
    rows = (await session.execute(select(Job).where(Job.source == SOURCE))).scalars().all()
    for j in rows:
        new_desc = _tidy(j.description)
        new_title = _tidy(j.title)
        if (new_desc, new_title) != (j.description, j.title):
            j.description, j.title = new_desc, new_title[:500]
            j.content_hash = content_hash(new_title, j.company, j.location, new_desc)
            j.embedding = None
            j.embedding_model = None
            fixed += 1
        if j.last_seen_at is not None and not looks_english(f"{j.title} {j.description}"):
            await session.execute(update(Job).where(Job.id == j.id).values(last_seen_at=None))
            closed += 1
    await session.commit()
    return {"rewritten": fixed, "closed_non_english": closed}
