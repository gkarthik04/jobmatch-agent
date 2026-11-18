"""Deterministic synthetic resumes (PDF) with known ground truth, for the structured-output eval.

15 resumes across 6 tracks and 3 seniority levels, with varied section names/orders. The ground
truth is what the generator wrote into the document, never what a model said.
Experience rule (same as the parser prompt): full-time months + 0.5 * internship months.
"""

import random
from dataclasses import dataclass, field
from pathlib import Path

OUT_DIR = Path("data/eval_resumes")

FIRST = ["Aarav", "Priya", "Rohan", "Ananya", "Vikram", "Meera", "Arjun", "Kavya", "Siddharth", "Neha",
         "Daniel", "Sofia", "Marcus", "Chloe", "Ethan"]
LAST = ["Sharma", "Iyer", "Reddy", "Nair", "Gupta", "Menon", "Patel", "Rao", "Das", "Kulkarni",
        "Brooks", "Nguyen", "Fischer", "Okafor", "Silva"]
UNIS = ["Anna University", "VIT Vellore", "NIT Trichy", "BITS Pilani", "IIIT Hyderabad", "PES University",
        "University of Texas at Austin", "Purdue University", "University of Waterloo", "RV College of Engineering"]
DEGREES = ["B.Tech in Computer Science", "B.E. in Information Technology", "B.Tech in Electronics and Communication",
           "B.S. in Computer Science", "M.S. in Data Science", "M.Tech in Artificial Intelligence"]
CITIES = ["Bangalore, India", "Hyderabad, India", "Chennai, India", "Pune, India", "Austin, TX", "Seattle, WA",
          "Toronto, Canada", "London, UK"]
COMPANIES = ["Infosys", "Zoho", "Freshworks", "Razorpay", "Swiggy", "Thoughtworks", "Mu Sigma", "Capgemini",
             "Acme Cloud", "Northwind Labs", "BrightPath", "Helix Analytics"]

TRACKS = {
    "backend": dict(role="Backend Engineer", skills=["Python", "Flask", "FastAPI", "PostgreSQL", "MySQL", "Redis",
                    "Docker", "REST APIs", "Git", "AWS"], titles=["Software Engineer", "Backend Developer"],
                    projects=[("URL Shortener Service", "Scalable URL shortener with Redis caching", ["FastAPI", "Redis"]),
                              ("Inventory API", "REST API for inventory management", ["Flask", "MySQL"]),
                              ("Task Queue Dashboard", "Monitoring UI for Celery workers", ["Python", "Docker"])]),
    "data_analyst": dict(role="Data Analyst", skills=["SQL", "Excel", "Tableau", "Power BI", "Python", "Pandas",
                         "Statistics", "A/B Testing"], titles=["Data Analyst", "Business Analyst"],
                         projects=[("Sales Dashboard", "Interactive revenue dashboard", ["Tableau", "SQL"]),
                                   ("Churn Analysis", "Customer churn drivers analysis", ["Python", "Pandas"]),
                                   ("Retail Forecasting", "Demand forecast for retail SKUs", ["Python", "Statistics"])]),
    "ml": dict(role="Machine Learning Engineer", skills=["Python", "PyTorch", "TensorFlow", "scikit-learn", "NLP",
               "Deep Learning", "SQL", "Docker", "Computer Vision"], titles=["ML Engineer", "Data Scientist"],
               projects=[("Resume Screening RAG Bot", "Retrieval-augmented QA over resumes", ["FAISS", "Sentence Transformers"]),
                         ("Image Defect Detector", "CNN defect detection for manufacturing", ["PyTorch", "OpenCV"]),
                         ("Sentiment Analyzer", "Fine-tuned BERT for product reviews", ["TensorFlow", "NLP"])]),
    "frontend": dict(role="Frontend Developer", skills=["JavaScript", "TypeScript", "React", "HTML", "CSS",
                     "Redux", "Git", "Figma"], titles=["Frontend Developer", "UI Engineer"],
                     projects=[("Portfolio Builder", "Drag-and-drop portfolio site generator", ["React", "TypeScript"]),
                               ("Weather App", "Responsive weather app using public APIs", ["JavaScript", "CSS"]),
                               ("Kanban Board", "Collaborative kanban with live updates", ["React", "Redux"])]),
    "devops": dict(role="DevOps Engineer", skills=["Docker", "Kubernetes", "Terraform", "AWS", "Linux", "Jenkins",
                   "GitHub Actions", "Python", "Bash"], titles=["DevOps Engineer", "Cloud Engineer"],
                   projects=[("CI/CD Pipeline", "End-to-end pipeline with automated deploys", ["Jenkins", "Docker"]),
                             ("IaC Landing Zone", "Terraform modules for AWS accounts", ["Terraform", "AWS"]),
                             ("Log Aggregator", "Centralised logging stack", ["Linux", "Kubernetes"])]),
    "fullstack": dict(role="Full Stack Developer", skills=["Java", "Spring Boot", "React", "SQL", "MongoDB",
                      "Docker", "Git", "REST APIs"], titles=["Full Stack Developer", "Software Developer"],
                      projects=[("Hospital Booking System", "Appointment booking platform", ["Spring Boot", "React"]),
                                ("Expense Tracker", "Personal finance tracker", ["Java", "MongoDB"]),
                                ("Chat Platform", "Real-time chat with websockets", ["Node.js", "React"])]),
}
LEVELS = {  # (full-time months list, internship months list)
    "student": ([], [2, 3]),
    "junior": ([14], [3]),
    "mid": ([30, 20], []),
}
DURATION_STYLES = ["dates_only", "dates_and_months"]


@dataclass
class Truth:
    name: str
    skills: list[str]
    experience_years: float
    education_institutions: list[str]
    project_names: list[str]
    objective_role: str | None
    stated_location: str | None
    path: str = ""
    track: str = ""
    level: str = ""
    extra: dict = field(default_factory=dict)


_MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _mon(idx: int) -> str:
    return f"{_MON[idx % 12]} {idx // 12}"


def _months_label(m: int) -> str:
    return f"{m} months" if m < 12 else (f"{m // 12} yr {m % 12} mos" if m % 12 else f"{m // 12} yrs")


def _build(i: int, rnd: random.Random) -> tuple[list[str], Truth]:
    track_key = list(TRACKS)[i % len(TRACKS)]
    level_key = list(LEVELS)[(i + i // 6) % len(LEVELS)]  # 15 unique (track, level) pairs
    t, (ft, intern) = TRACKS[track_key], LEVELS[level_key]
    name = f"{FIRST[i]} {LAST[(i * 7) % len(LAST)]}"
    city = CITIES[i % len(CITIES)]
    uni = UNIS[(i * 3) % len(UNIS)]
    degree = DEGREES[(i * 5) % len(DEGREES)]
    year = 2025 if level_key == "student" else 2023 - (i % 3)
    skills = rnd.sample(t["skills"], k=min(len(t["skills"]), rnd.randint(6, 8)))
    projects = rnd.sample(t["projects"], k=2)
    objective = i % 3 != 2  # a third of resumes state no objective/location in the summary
    style = DURATION_STYLES[i % 2]
    skills_head = ["Skills", "Technical Skills", "Core Competencies"][i % 3]

    lines = [name, f"{name.lower().replace(' ', '.')}@example.com | +91 98765 4{i:04d}"]
    stated_loc = None
    if objective:
        lines.append(f"Objective: Seeking {t['role']} roles. Based in {city}.")
        stated_loc = city
    else:
        lines.append("Summary: Motivated engineer who enjoys building reliable software.")
    lines += ["", "EDUCATION", f"{degree}, {uni}, {year}"]

    exp_months = 0.0
    lines += ["", "EXPERIENCE"]
    cos = rnd.sample(COMPANIES, 3)
    for k, m in enumerate(ft):
        exp_months += m
        start = (2020 + k * 2) * 12 + k  # month index; dates and stated duration always agree
        dur = f" ({_months_label(m)})" if style == "dates_and_months" else ""
        lines.append(f"{t['titles'][k % 2]}, {cos[k]} - {_mon(start)} to {_mon(start + m)}{dur}")
        lines.append("  - Built and maintained production services used by thousands of users")
    for k, m in enumerate(intern):
        exp_months += 0.5 * m
        start = (2022 + k) * 12 + 4  # May
        dur = f" ({_months_label(m)})" if style == "dates_and_months" else ""
        lines.append(f"{t['titles'][0]} Intern, {cos[(k + 1) % 3]} - {_mon(start)} to {_mon(start + m)}{dur}")
        lines.append("  - Contributed features and fixed bugs under a senior mentor")
    if not ft and not intern:
        lines.append("No professional experience yet.")

    lines += ["", "PROJECTS"]
    for pn, pd, tech in projects:
        lines.append(f"{pn}: {pd}. Tech: {', '.join(tech)}")
    lines += ["", skills_head.upper(), ", ".join(skills)]
    truth = Truth(
        name=name, skills=skills, experience_years=round(exp_months / 12, 2),
        education_institutions=[uni], project_names=[p[0] for p in projects],
        objective_role=t["role"] if objective else None, stated_location=stated_loc,
        track=track_key, level=level_key,
    )
    return lines, truth


def make_pdf(lines: list[str]) -> bytes:
    import io

    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    y = 800
    for ln in lines:
        c.setFont("Helvetica-Bold" if ln.isupper() and ln else "Helvetica", 10)
        c.drawString(40, y, ln[:110])
        y -= 14
    c.save()
    return buf.getvalue()


def generate(n: int = 15, seed: int = 7) -> list[tuple[bytes, Truth]]:
    rnd = random.Random(seed)
    out = []
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        lines, truth = _build(i, rnd)
        pdf = make_pdf(lines)
        truth.path = str(OUT_DIR / f"resume_{i + 1:02d}.pdf")
        Path(truth.path).write_bytes(pdf)
        out.append((pdf, truth))
    return out
