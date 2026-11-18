"""What counts as an open position, and where its apply link points."""

from datetime import UTC, datetime, timedelta

from app.services.live_feed import OPEN_WINDOW_DAYS

ARCHIVE_SOURCE = "linkedin_2024"
SOURCE_LABELS = {"arbeitnow": "Arbeitnow", ARCHIVE_SOURCE: "LinkedIn, April 2024"}


def is_open(job, now: datetime | None = None) -> bool:
    """Open = from a live feed, seen in the latest fetches, and not past an expiry date.

    Archived dataset postings are never open: their recorded expiry dates are all in May 2024.
    """
    now = now or datetime.now(UTC)
    if job.expires_at is not None and job.expires_at < now:
        return False
    if (job.source or ARCHIVE_SOURCE) == ARCHIVE_SOURCE:
        return False
    seen = job.last_seen_at
    return seen is not None and now - seen < timedelta(days=OPEN_WINDOW_DAYS)


def source_label(source: str | None) -> str:
    return SOURCE_LABELS.get(source or ARCHIVE_SOURCE, source or "Unknown")


def link_for(job) -> str | None:
    return job.apply_url or job.url


# SQL twin of is_open() for filtering inside the database
OPEN_SQL = (
    f"source <> '{ARCHIVE_SOURCE}' AND last_seen_at > now() - interval '{OPEN_WINDOW_DAYS} days' "
    "AND (expires_at IS NULL OR expires_at > now())"
)
