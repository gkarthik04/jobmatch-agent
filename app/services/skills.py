"""Lexicon-based skill extraction. Deterministic, free, and good enough for a retrieval index."""

import re

# canonical name -> regex alternatives (matched case-insensitively on word boundaries)
_LEXICON: dict[str, list[str]] = {
    "python": ["python"], "java": ["java"], "javascript": ["javascript", "js"],
    "typescript": ["typescript"], "c++": [r"c\+\+"], "c#": [r"c#"], "go": [r"golang"],
    "rust": ["rust"], "scala": ["scala"], "kotlin": ["kotlin"], "swift": ["swift"],
    "ruby": ["ruby"], "php": ["php"], "r": [r"r programming", r"rstudio"],
    "sql": ["sql", "t-sql", "pl/sql"], "postgresql": ["postgres", "postgresql"],
    "mysql": ["mysql"], "mongodb": ["mongodb"], "redis": ["redis"], "snowflake": ["snowflake"],
    "spark": ["spark", "pyspark"], "hadoop": ["hadoop"], "kafka": ["kafka"],
    "airflow": ["airflow"], "dbt": ["dbt"], "etl": ["etl"], "bigquery": ["bigquery"],
    "react": ["react", "react.js", "reactjs"], "angular": ["angular"], "vue": ["vue", "vue.js"],
    "node.js": ["node.js", "nodejs", "node"], "django": ["django"], "flask": ["flask"],
    "fastapi": ["fastapi"], "spring": ["spring boot", "spring"], ".net": [r"\.net", "dotnet"],
    "html": ["html"], "css": ["css"], "graphql": ["graphql"], "rest apis": ["rest", "restful"],
    "docker": ["docker"], "kubernetes": ["kubernetes", "k8s"], "terraform": ["terraform"],
    "aws": ["aws", "amazon web services"], "azure": ["azure"],
    "gcp": ["gcp", "google cloud"], "linux": ["linux"], "git": ["git", "github"],
    "ci/cd": [r"ci/cd", "jenkins", "github actions"], "devops": ["devops"],
    "machine learning": ["machine learning", "ml"], "deep learning": ["deep learning"],
    "nlp": ["nlp", "natural language processing"], "llm": ["llm", "llms", "large language model"],
    "pytorch": ["pytorch"], "tensorflow": ["tensorflow"], "scikit-learn": ["scikit-learn", "sklearn"],
    "pandas": ["pandas"], "numpy": ["numpy"], "computer vision": ["computer vision"],
    "data analysis": ["data analysis", "data analytics"], "statistics": ["statistics", "statistical"],
    "tableau": ["tableau"], "power bi": ["power bi", "powerbi"], "excel": ["excel"],
    "looker": ["looker"], "a/b testing": [r"a/b test\w*", "experimentation"],
    "data engineering": ["data engineering", "data pipelines"],
    "microservices": ["microservices"], "agile": ["agile", "scrum"], "jira": ["jira"],
    "cybersecurity": ["cybersecurity", "information security", "infosec"],
    "networking": ["networking", "tcp/ip"], "embedded": ["embedded", "firmware"],
    "android": ["android"], "ios": ["ios"], "unity": ["unity"], "selenium": ["selenium"],
    "testing": ["unit testing", "test automation", "qa"], "salesforce": ["salesforce"],
    "sap": ["sap"], "figma": ["figma"], "ux design": ["ux", "user experience"],
    "product management": ["product management", "roadmap"], "seo": ["seo"],
    "project management": ["project management", "pmp"], "customer service": ["customer service"],
    "sales": ["sales", "business development"], "marketing": ["marketing"],
    "accounting": ["accounting", "gaap", "bookkeeping"], "finance": ["financial analysis", "fp&a"],
    "nursing": ["rn", "registered nurse", "bls", "acls"], "communication": ["communication skills"],
    "leadership": ["leadership", "team management"], "hr": ["recruiting", "human resources"],
}

_PATTERNS = {
    name: re.compile(r"(?<![\w+#.])(?:" + "|".join(alts) + r")(?![\w+#]|\.\w)", re.IGNORECASE)
    for name, alts in _LEXICON.items()
}


def extract_skills(text: str, limit: int = 25) -> list[str]:
    found = [name for name, pat in _PATTERNS.items() if pat.search(text or "")]
    return found[:limit]


def all_skill_names() -> list[str]:
    return list(_LEXICON)
