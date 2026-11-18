"""Golden retrieval queries.

Each query is a natural-language request written the way a job seeker would phrase it (deliberately
NOT copying posting titles, so pure keyword matching is not trivially favoured). Relevance is
defined by an explicit, versioned rubric: a case-insensitive regex that a relevant posting's TITLE
must match (optionally also a regex the title must NOT match). `build_golden.py` applies the
rubric to a frozen snapshot of the index and writes the resulting relevant source_ids to
`golden.json`, so labels are fixed and independent of any retriever being evaluated.
"""

# (query, title_must_match, title_must_not_match | None)
QUERIES: list[tuple[str, str, str | None]] = [
    # --- data / ML ---
    ("I want to build and maintain data pipelines and ETL workflows",
     r"data engineer|\betl\b|data pipeline|analytics engineer|big data", None),
    ("analyse business data and build dashboards for stakeholders",
     r"data analyst|business intelligence|\bbi\b (analyst|developer|engineer)|reporting analyst|analytics", None),
    ("train and deploy machine learning models",
     r"machine learning|data scientist|\bai\b|\bml\b|applied scientist", None),
    ("statistical modelling and experimentation role", r"data scientist|statistic|quantitative|research scientist", None),
    # --- software ---
    ("write backend services and REST APIs",
     r"back.?end|python|java (developer|engineer)|software (engineer|developer)|api", None),
    ("build modern user interfaces with React",
     r"front.?end|react|\bui\b (engineer|developer)|web developer", None),
    ("full stack web developer who can do both client and server",
     r"full.?stack", None),
    ("cloud infrastructure and CI/CD automation",
     r"devops|cloud|site reliability|\bsre\b|platform engineer|infrastructure", None),
    ("automate tests and ensure software quality",
     r"\bqa\b|quality assurance|sdet|test (engineer|automation|analyst)|software tester",
     r"pharma|regulatory|clinical|food|lab\b|manufactur|quality assurance (associate|specialist|inspector)"),
    ("mobile app development for iOS or Android", r"\bios\b|android|mobile", None),
    ("entry level software engineering role for a new graduate",
     r"(intern|new grad|entry.level|junior|graduate|associate).*(software|engineer|developer)|"
     r"(software|engineer|developer).*(intern|entry.level|junior|associate)", None),
    ("lead an engineering team", r"engineering manager|director of engineering|tech(nical)? lead|development manager", None),
    # --- security / IT ---
    ("protect company systems from cyber threats", r"security|cyber|infosec|soc analyst",
     r"security (officer|guard)|social security|loss prevention"),
    ("keep company computers and networks running smoothly",
     r"network|system administrator|sysadmin|it support|help ?desk|desktop support|technical support", None),
    # --- product / design / analysis ---
    ("own the product roadmap and work with engineers", r"product (manager|owner|management)", None),
    ("design user experiences and interfaces",
     r"\bux\b|\bui\b|user experience|product designer|interaction design|visual design|web design",
     r"developer|engineer"),
    ("manage technology or software delivery projects",
     r"(project|program) manager|scrum master|technical program", None),
    ("gather business requirements and translate them for developers",
     r"business analyst|systems analyst|requirements", None),
    # --- business ---
    ("financial analysis, budgeting and forecasting", r"financial analyst|fp&a|finance|budget", None),
    ("bookkeeping, accounts payable and receivable", r"accountant|accounting|bookkeep|accounts (payable|receivable)|auditor", None),
    ("inside sales and business development for B2B clients",
     r"sales (representative|associate|executive|development)|account executive|business development|inside sales", None),
    ("run digital marketing campaigns and SEO", r"marketing|\bseo\b|social media|content", None),
    ("recruiting and talent acquisition", r"recruit|talent|human resources|\bhr\b", None),
    ("help customers over phone and chat", r"customer (service|support|success)|call cent|client service", None),
    ("supply chain, procurement and logistics coordination",
     r"supply chain|logistics|procurement|buyer|purchasing", None),
    ("administrative and office support", r"administrative|office manager|receptionist|executive assistant|secretary", None),
    # --- other sectors ---
    ("teach students in a classroom", r"teacher|instructor|professor|tutor|educator|faculty", None),
    ("patient care in a hospital or clinic",
     r"nurs|\brn\b|\blpn\b|medical|clinical|patient|health ?care|therapist|physician|caregiver", None),
    ("work in a pharmacy", r"pharmac", None),
    ("drive trucks or make deliveries", r"driver|\bcdl\b|courier|delivery (associate|specialist)", r"software|engineer|manager"),
    ("warehouse and forklift work", r"warehouse|forklift|material handler|picker", None),
    ("mechanical engineering and product design", r"mechanical|manufacturing engineer|design engineer",
     r"software|electrical|hardware|technician"),
    ("electrical or electronics engineering", r"electrical|electronic|embedded|hardware engineer", None),
    ("civil engineering and construction projects",
     r"civil|construction|structural|project engineer|superintendent", None),
    ("manage a retail store", r"store manager|retail|assistant manager|shift (lead|supervisor)", None),
    ("restaurant and hospitality work", r"restaurant|\bcook\b|chef|server|hospitality|hotel|barista|bartender",
     r"engineer|administrator|software|systems|network"),
    ("legal work at a law firm", r"attorney|paralegal|legal|counsel|lawyer", None),
    ("real estate and property management", r"real estate|realtor|leasing|property manager", None),
    ("insurance underwriting and claims", r"insurance|underwrit|claims", None),
    ("banking, loans and mortgage", r"bank|teller|loan|mortgage|credit", None),
    ("equipment maintenance technician", r"technician|mechanic|maintenance",
     r"software|data|network|\bit\b|pharmacy|dental|lab|medical|radiolog|surgical|vet|nurs|pharm|sterile"),
    ("laboratory and scientific research", r"scientist|laborator|research|chemist|biolog",
     r"data scientist|market|survey|user research"),
    ("data entry and clerical work", r"data entry|clerk", None),
    ("write and edit content", r"writer|editor|copywriter|journalist|content", None),
]
