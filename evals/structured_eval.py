"""Structured-output eval: resume PDF -> ProfileSchema on 15 synthetic resumes with known truth."""

import re

from app.errors import AppError
from app.schemas import ProfileSchema
from app.services.resume import extract_pdf_text, parse_resume
from evals.metrics import mean
from evals.synth_resumes import Truth, generate


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9+#]", "", s.lower())


def same_skill(a: str, b: str) -> bool:
    a, b = norm(a), norm(b)
    if not a or not b:
        return False
    return a == b or (min(len(a), len(b)) >= 3 and (a.startswith(b.rstrip("s")) or b.startswith(a.rstrip("s"))))


def skill_prf(pred: list[str], truth: list[str]) -> tuple[float, float, float]:
    tp_r = sum(any(same_skill(t, p) for p in pred) for t in truth)
    tp_p = sum(any(same_skill(p, t) for t in truth) for p in pred)
    r = tp_r / len(truth) if truth else 1.0
    p = tp_p / len(pred) if pred else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def _tokens(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z]+", s.lower()) if len(t) > 2 and t not in {"and", "the", "roles", "role"}}


def score_profile(p: ProfileSchema, t: Truth) -> dict[str, float | None]:
    prec, rec, f1 = skill_prf(p.skills, t.skills)
    pred_insts = " ".join(f"{e.institution} {e.degree}" for e in p.education).lower()
    proj_blob = " ".join(x.name for x in p.projects).lower()
    roles_tokens = set().union(*[_tokens(r) for r in p.preferred_roles]) if p.preferred_roles else set()
    return {
        "name_exact": float(norm(p.name) == norm(t.name)),
        "skills_precision": prec, "skills_recall": rec, "skills_f1": f1,
        "experience_within_0.5y": float(abs(p.experience_years - t.experience_years) <= 0.5),
        "experience_abs_err": abs(p.experience_years - t.experience_years),
        "education_institution": float(all(i.lower() in pred_insts for i in t.education_institutions)),
        "projects_recall": mean([float(n.lower() in proj_blob) for n in t.project_names]),
        # only scored when the resume states an objective / a location:
        "preferred_role": None if t.objective_role is None else float(bool(_tokens(t.objective_role) & roles_tokens)),
        "location_stated": None if t.stated_location is None else float(
            any(_tokens(t.stated_location) & _tokens(loc) for loc in p.preferred_locations)),
        "no_invented_location": None if t.stated_location is not None else float(not p.preferred_locations),
    }


async def run(llm, n: int = 15, system: str | None = None) -> dict:
    rows, valid_first, valid_final = [], 0, 0
    for pdf, truth in generate(n):
        text = extract_pdf_text(pdf)
        stats: dict = {}
        try:
            profile = await parse_resume(llm, text, stats=stats, system=system)
        except AppError:
            rows.append({"resume": truth.path, "valid": False, "attempts": stats.get("attempts", 0)})
            continue
        valid_final += 1
        valid_first += stats["attempts"] == 1
        rows.append({"resume": truth.path, "valid": True, "attempts": stats["attempts"],
                     "track": truth.track, "level": truth.level, **score_profile(profile, truth)})
    scored = [r for r in rows if r["valid"]]
    fields = [k for k in (scored[0] if scored else {}) if k not in ("resume", "valid", "attempts", "track", "level")]
    agg = {}
    for k in fields:
        vals = [r[k] for r in scored if r[k] is not None]
        agg[k] = (mean(vals), len(vals))
    return {
        "n": n, "valid_first_try": valid_first / n, "valid_after_retry": valid_final / n,
        "fields": agg, "rows": rows,
    }
