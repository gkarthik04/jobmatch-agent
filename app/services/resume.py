"""Resume PDF -> validated ProfileSchema. Structured output with validate -> retry-with-error -> fail."""

import io
import logging

from pydantic import ValidationError

from app.errors import AppError
from app.schemas import ProfileSchema
from app.services.llm import LLMUnavailable

log = logging.getLogger("app.resume")

MAX_BYTES = 5 * 1024 * 1024
MAX_PAGES = 6
MAX_CHARS = 12_000
MAX_ATTEMPTS = 3

SYSTEM_V1 = (
    "You extract structured data from resumes. Return ONLY JSON matching the schema. "
    "Rules: skills are short canonical names (e.g. 'Python', 'PostgreSQL'), deduplicated, max 40. "
    "experience_years is total professional experience as a number (internships count half; "
    "0 for students with no experience; never a range). education lists degrees with the graduation "
    "year as an integer if present else null. projects: name, one-sentence description, technologies. "
    "preferred_roles: 1-4 job titles the candidate is best suited for or targets (infer from the "
    "resume if no objective is stated). preferred_locations: locations stated or implied by "
    "the resume, [] if none. Do not invent facts that are not supported by the resume."
)


# v2 = v1 + an explicit month-by-month procedure for experience_years. Measured in
# evals/RESULTS.md it did NOT beat v1 (86.7% vs 93.3% within 0.5y on 15 resumes), so v1 ships.
SYSTEM_V2 = SYSTEM_V1 + (
    " experience_years procedure: for every job/internship take (end date - start date) in months "
    "from the DATES (if a stated duration contradicts the dates, trust the dates); full-time roles "
    "count 1x, internships/trainee roles count 0.5x; never count projects or education; do not "
    "double count overlapping periods; sum the months, divide by 12, round to 1 decimal."
)
PROMPTS = {"v1": SYSTEM_V1, "v2": SYSTEM_V2}
SYSTEM = SYSTEM_V1


def extract_pdf_text(data: bytes) -> str:
    if not data.startswith(b"%PDF"):
        raise AppError(415, "not_a_pdf", "File is not a PDF")
    if len(data) > MAX_BYTES:
        raise AppError(413, "file_too_large", "PDF is larger than 5 MB")
    from pypdf import PdfReader
    from pypdf.errors import PyPdfError

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise AppError(422, "pdf_encrypted", "PDF is password protected")
        pages = [(p.extract_text() or "") for p in reader.pages[:MAX_PAGES]]
    except PyPdfError as exc:
        raise AppError(422, "pdf_unreadable", "Could not read this PDF") from exc
    text = "\n".join(pages).strip()
    if len(text) < 80:
        raise AppError(422, "pdf_no_text", "No extractable text found (scanned PDFs are not supported)")
    return text[:MAX_CHARS]


async def parse_resume(
    llm, text: str, stats: dict | None = None, system: str | None = None
) -> ProfileSchema:
    """Ask the LLM for JSON, validate with Pydantic; on failure feed the error back and retry."""
    prompt = f"Resume text:\n\"\"\"\n{text}\n\"\"\""
    last_error = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        contents = prompt
        if last_error:
            contents += (
                "\n\nYour previous answer was invalid. Fix these problems and return corrected JSON "
                f"only:\n{last_error}"
            )
        try:
            res = await llm.generate(contents, system=system or SYSTEM, schema=ProfileSchema)
            profile = ProfileSchema.model_validate_json(res.text)
            if not 0 <= profile.experience_years <= 60:
                raise ValueError("experience_years must be between 0 and 60")
            if not profile.name.strip() or not profile.skills:
                raise ValueError("name and at least one skill are required")
            profile.skills = list(dict.fromkeys(s.strip() for s in profile.skills if s.strip()))[:40]
            if stats is not None:
                stats["attempts"] = attempt
            return profile
        except LLMUnavailable as exc:
            raise AppError(503, "llm_unavailable",
                           "The language model is busy right now, please retry in a minute") from exc
        except (ValidationError, ValueError) as exc:
            last_error = str(exc)[:800]
            log.warning("resume parse invalid", extra={"attempt": attempt, "error": last_error})
    if stats is not None:
        stats["attempts"] = MAX_ATTEMPTS
    raise AppError(
        422, "resume_parse_failed", "Could not extract a valid profile from this resume",
        details={"last_error": last_error},
    )
