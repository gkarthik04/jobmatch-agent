"""CLI: data fetch, incremental ingestion and embedding.

    python cli.py fetch                 # download the dataset sample (first ~60MB)
    python cli.py ingest --add 4000     # initial load (oldest postings first)
    python cli.py ingest --add 200      # incremental: next 200 postings + any changed rows
    python cli.py stats
"""

import argparse
import asyncio
import sys

from app.db import dispose_engine, get_sessionmaker
from app.logging_conf import setup_logging


def fetch() -> None:
    import urllib.request

    from app.services.ingest import DATA_URL, HEAD_BYTES, RAW_PATH

    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    if RAW_PATH.exists() and RAW_PATH.stat().st_size > HEAD_BYTES * 0.9:
        print(f"{RAW_PATH} already present")
        return
    req = urllib.request.Request(DATA_URL, headers={"Range": f"bytes=0-{HEAD_BYTES}"})
    with urllib.request.urlopen(req, timeout=300) as r, open(RAW_PATH, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    print(f"downloaded {RAW_PATH.stat().st_size / 1e6:.1f} MB")


async def ingest(add: int, no_embed: bool) -> None:
    from app.services import ingest as ing
    from app.services.embeddings import make_embedder

    feed = ing.load_feed()
    print(f"feed: {len(feed)} clean postings")
    async with get_sessionmaker()() as session:
        stats = await ing.upsert_jobs(session, feed, add_new=add)
        print(f"upsert: {stats}")
        if not no_embed:
            stats.embedded = await ing.embed_pending(session, make_embedder())
            print(f"embedded {stats.embedded} rows")
            await ing.analyze(session)
    await dispose_engine()


async def stats() -> None:
    from sqlalchemy import text

    async with get_sessionmaker()() as s:
        row = (await s.execute(text(
            "select count(*), count(embedding), min(posted_at), max(posted_at) from jobs"))).one()
    print(f"jobs={row[0]} embedded={row[1]} posted {row[2]} .. {row[3]}")
    await dispose_engine()


def main() -> None:
    setup_logging("INFO")
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch")
    sub.add_parser("stats")
    pi = sub.add_parser("ingest")
    pi.add_argument("--add", type=int, default=200, help="max NEW postings to add this run")
    pi.add_argument("--no-embed", action="store_true")
    a = p.parse_args()
    if a.cmd == "fetch":
        fetch()
    elif a.cmd == "ingest":
        asyncio.run(ingest(a.add, a.no_embed))
    elif a.cmd == "stats":
        asyncio.run(stats())


if __name__ == "__main__":
    sys.exit(main())
