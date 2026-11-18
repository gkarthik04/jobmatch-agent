"""Live job feed: currently open positions with real apply links.

Source: the Arbeitnow public job board API (free, no key, documented for external use; mostly
Europe). Every job keeps its Arbeitnow URL as the apply link and is shown with a "via Arbeitnow"
credit. We fetch a few pages once a day; `last_seen_at` marks a job as still open, and a job that
stops appearing in the feed is treated as closed after OPEN_WINDOW_DAYS.
"""

import hashlib
import json
import logging
import re
import time
import urllib.request
from datetime import UTC, datetime

from app.services.ingest import MIN_DESC_CHARS, CleanJob, clean_text, content_hash, normalize_link
from app.services.skills import extract_skills

log = logging.getLogger("app.live_feed")

ARBEITNOW_URL = "https://www.arbeitnow.com/api/job-board-api"
USER_AGENT = "jobmatch-portfolio/1.0 (open source demo; contact via GitHub)"
SOURCE = "arbeitnow"
OPEN_WINDOW_DAYS = 3

_EN = {"the", "and", "you", "we", "with", "for", "our", "will", "your", "to", "of", "in", "is", "are"}
_DE = {"und", "der", "die", "das", "wir", "für", "mit", "sie", "ein", "eine", "ist", "den", "von", "zu"}


_PROMO = re.compile(r"\s*Find (more )?(English Speaking )?Jobs in [A-Za-z ]+ on Arbeitnow\s*$", re.I)


def make_source_id(slug: str) -> str:
    """Stable short id: slugs can exceed the 64 character source_id column."""
    return f"{SOURCE}:{hashlib.sha1(slug.encode()).hexdigest()[:24]}"


def _tidy(s: str) -> str:
    """The feed contains U+FFFD replacement characters (a broken dash) and a promo footer."""
    return _PROMO.sub("", clean_text(s).replace("\ufffd", "-")).strip()


def looks_english(text_: str) -> bool:
    """Cheap stopword vote. The index uses an English embedding model, so we skip German postings."""
    words = re.findall(r"[a-zäöüß]+", text_.lower())[:400]
    en = sum(w in _EN for w in words)
    de = sum(w in _DE for w in words)
    return en >= 3 and en > de


def fetch_pages(pages: int = 3, timeout: float = 30.0, delay_s: float = 1.0) -> list[dict]:
    """GET up to `pages` pages from the feed (blocking; call from a thread)."""
    jobs: list[dict] = []
    for page in range(1, pages + 1):
        req = urllib.request.Request(f"{ARBEITNOW_URL}?page={page}", headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - fixed https URL
            data = json.loads(r.read().decode("utf-8"))
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
