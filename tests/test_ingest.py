from app.services.ingest import clean_row, clean_text, content_hash
from app.services.skills import extract_skills

DESC = "We build data pipelines in Python and SQL on AWS with Docker. " * 6


def row(**kw):
    base = {"job_id": 123.0, "title": "Data Engineer", "company_name": "Acme",
            "description": DESC, "location": "Austin, TX", "listed_time": 1712345678000.0,
            "remote_allowed": float("nan")}
    return {**base, **kw}


def test_clean_text_strips_html_and_entities():
    assert clean_text("<p>Hello&nbsp;<b>world</b> &amp; co</p>").split() == ["Hello", "world", "&", "co"]


def test_clean_row_basic():
    j = clean_row(row())
    assert j and j.source_id == "123" and j.remote is False
    assert j.posted_at and j.posted_at.year == 2024
    assert {"python", "sql", "aws", "docker"} <= set(j.skills)


def test_clean_row_drops_short_descriptions():
    assert clean_row(row(description="too short")) is None


def test_remote_detected_from_flag_or_title():
    assert clean_row(row(remote_allowed=1.0)).remote is True
    assert clean_row(row(title="Backend Engineer (Remote)")).remote is True


def test_content_hash_changes_with_description():
    assert content_hash("a", "b", "c", "d") != content_hash("a", "b", "c", "e")


def test_skill_extraction_boundaries():
    s = extract_skills("Experience with MySQL, C++ and JavaScript; not just java-like things")
    assert "mysql" in s and "c++" in s and "javascript" in s
    assert "sql" not in s  # 'sql' inside 'mysql' must not match
