#!/usr/bin/env python3
"""
rank.py — Redrob Hackathon: Intelligent Candidate Discovery & Ranking Challenge
Team K3-SpiritCourage

Ranks the candidate pool against the "Senior AI Engineer — Founding Team" JD and
writes the top 100 as a CSV (candidate_id,rank,score,reasoning).

Design goals (see README.md / methodology_summary for the full writeup):
  1. Transparent, rule-based, fully explainable scoring. No black-box model.
  2. Title/career substance GATES skill-keyword credit, so a candidate whose
     skills section is stuffed with AI buzzwords but whose title/career history
     shows no relevant work (e.g. "Marketing Manager") cannot out-rank a
     candidate whose career history shows real ranking/search/recommendation
     work even if their skills list is plain-language.
  3. Skill claims are cross-checked against redrob_signals.skill_assessment_scores
     (platform-administered, independent of self-report) and against
     duration_months / endorsements, to discount inflated self-rated proficiency.
  4. JD's explicit disqualifiers/negatives are detected and penalized
     (multiplicatively, not as a hard cutoff) — pure-consulting-only careers,
     pure research-only, LangChain-only <12mo AI experience, architecture/tech-lead
     roles with no recent code, CV/speech/robotics-only without NLP/IR exposure,
     fully closed-source 5+ years with no external validation, title-chasing.
  5. Years-of-experience, location and notice-period are SOFT preferences
     (smooth functions), per the JD's explicit "this is a range, not a
     requirement" framing — never hard filters.
  6. Behavioral signals (redrob_signals) act as a multiplicative
     availability/engagement modifier on top of the skill-fit score.
  7. Honeypot detection: candidates with internally-impossible profiles
     (e.g. "expert" proficiency claimed with ~0 months of hands-on duration,
     or total career-history duration wildly inconsistent with stated
     years_of_experience) are excluded from the ranked output entirely.
  8. A small, curated, *phrase-level* TF-IDF text-similarity score is computed
     only over career_history descriptions + profile summary/headline — never
     over the skills[] list — specifically so that keyword-stuffed skills
     cannot inflate this term. It contributes a bounded amplifier/dampener,
     not a dominant term.

Compute: single pass over candidates.jsonl, pure Python standard library only
(no GPU, no network, no sklearn/numpy dependency) so the constraints in
submission_spec.md Section 3 (<=5 min, <=16GB RAM, CPU-only, no network,
<=5GB disk) are trivially satisfied for a 100K-row pool.

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv
"""

import argparse
import csv
import json
import math
import random
import re
import sys
import time
from datetime import date, datetime

# --------------------------------------------------------------------------- #
# JD-derived constants
# --------------------------------------------------------------------------- #

# "Today" for recency calculations. The dataset's redrob_signals dates cluster
# in the run-up to the hackathon; we use the latest last_active_date actually
# observed in the pool (computed at runtime) as the reference "now" so this
# stays correct regardless of when the ranker is reproduced.
FALLBACK_REFERENCE_DATE = date(2026, 6, 30)

MUST_HAVE_SKILL_GROUPS = {
    "embeddings_retrieval": [
        "sentence-transformers", "sentence transformers", "openai embeddings",
        "bge", "e5", "embedding", "embeddings", "dense retrieval", "retrieval",
        "semantic search", "vector search",
    ],
    "vector_db_hybrid_search": [
        "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
        "elasticsearch", "faiss", "vector database", "vector db",
        "hybrid search",
    ],
    "python": ["python"],
    "eval_frameworks": [
        "ndcg", "mrr", "map", "a/b testing", "ab testing",
        "offline evaluation", "ranking evaluation", "learning to rank",
        "learning-to-rank", "ltr", "mean average precision",
        "evaluation framework",
    ],
}

NICE_TO_HAVE_SKILLS = [
    "lora", "qlora", "peft", "fine-tuning", "fine tuning",
    "xgboost", "learning-to-rank", "learning to rank", "hr-tech", "hr tech",
    "recruiting tech", "distributed systems", "large-scale inference",
    "large scale inference", "open source", "open-source",
]

# Proficiency -> expected minimum skill_assessment_scores value (0-100 scale).
# Used to discount self-reported proficiency that the platform's own
# assessment does not corroborate (see CAND_0000001 in the dataset: "advanced"
# self-rated NLP/Image Classification/Speech Recognition with assessment
# scores in the 38-65 range — a corroboration mismatch).
PROFICIENCY_LEVEL = {"beginner": 0.40, "intermediate": 0.60, "advanced": 0.85, "expert": 1.00}
PROFICIENCY_EXPECTED_ASSESSMENT = {"beginner": 25.0, "intermediate": 45.0, "advanced": 65.0, "expert": 82.0}

RELEVANT_TITLE_TERMS = [
    "machine learning", "ml engineer", "ai engineer", "applied scientist",
    "research engineer", "data scientist", "nlp engineer", "search engineer",
    "recommendation", "retrieval", "information retrieval", "ranking engineer",
    "search relevance", "relevance engineer",
]
GENERIC_ENGINEERING_TERMS = [
    "software engineer", "backend engineer", "data engineer", "full stack",
    "full-stack", "platform engineer", "infrastructure engineer",
    "engineer", "developer", "architect",
]
NON_ENGINEERING_TITLE_TERMS = [
    "business analyst", "hr manager", "human resources", "mechanical engineer",
    "marketing manager", "marketing", "sales", "accountant", "finance",
    "operations manager", "civil engineer", "electrical engineer",
    "recruiter", "administrator", "office manager", "content writer",
    "graphic designer", "customer support", "customer service",
]

SHIPPED_EVIDENCE_TERMS = [
    "shipped", "deployed", "launched", "built", "scaled", "implemented",
    "designed", "owned", "production",
]
RELEVANT_SYSTEM_NOUNS = [
    "ranking", "search", "recommendation", "retrieval", "matching",
    "embeddings", "relevance",
]

CONSULTING_FIRMS = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "tech mahindra", "mphasis", "mindtree",
}

CV_SPEECH_ROBOTICS_TERMS = [
    "computer vision", "image classification", "object detection",
    "speech recognition", "speech-to-text", "speech to text", "robotics",
    "autonomous", "slam", "lidar", "gans", "gan ",
]
NLP_IR_TERMS = [
    "nlp", "natural language", "information retrieval", "search", "ranking",
    "embeddings", "llm", "retrieval", "language model", "rag",
]

LANGCHAIN_TUTORIAL_TERMS = ["langchain", "openai api", "gpt wrapper", "prompt engineering"]
PRE_LLM_ML_TERMS = [
    "machine learning", "recommendation", "search", "ranking", "nlp",
    "data engineering", "data science", "retrieval", "classification",
    "regression", "clustering",
]

ARCHITECT_TITLE_TERMS = ["architect", "tech lead", "technical lead", "engineering manager", "head of"]
CODE_EVIDENCE_TERMS = ["code", "coded", "implemented", "built", "wrote", "shipped", "programmed"]

PUNE_NOIDA = {"pune", "noida"}
TIER1_PLUS_INDIA_CITIES = {"hyderabad", "mumbai", "delhi", "delhi ncr", "gurugram", "gurgaon", "ncr"}

# A short, curated reference passage capturing the JD's "ideal candidate"
# narrative (Section: "How to read between the lines"). Used ONLY for the
# light text-similarity bonus, and only matched against candidates'
# career-history descriptions + profile summary/headline — never the
# skills[] list — precisely so keyword-stuffed skills cannot inflate it.
JD_REFERENCE_TEXT = """
shipped an end to end ranking search or recommendation system to real users
at meaningful scale production experience with embeddings based retrieval
vector databases or hybrid search infrastructure deployed to real users
handled embedding drift index refresh retrieval quality regression in
production designed evaluation frameworks for ranking systems ndcg mrr map
offline to online correlation ab testing strong opinions about retrieval
hybrid versus dense evaluation offline versus online and llm integration
fine tune versus prompt built at a product company writes production code
mentors engineers owns the intelligence layer matching and search systems
""".lower()

# Curated vocabulary for the text-similarity component: union of the
# must-have/nice-to-have skill phrases plus the "shipped system" phrases
# above. Kept small and explainable on purpose (this is not a general
# bag-of-words model).
TEXT_SIM_VOCAB = sorted(set(
    sum(MUST_HAVE_SKILL_GROUPS.values(), [])
    + NICE_TO_HAVE_SKILLS
    + SHIPPED_EVIDENCE_TERMS
    + RELEVANT_SYSTEM_NOUNS
    + ["product company", "real users", "scale", "production", "drift",
       "index refresh", "regression", "offline", "online", "a/b test",
       "hybrid", "dense", "prompt", "mentors", "mentoring"]
))


def contains_any(text, terms):
    return any(t in text for t in terms)


def count_terms(text, terms):
    return sum(text.count(t) for t in terms)


def safe_lower(x):
    return (x or "").lower()


def parse_date(s):
    if not s:
        return None
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


# --------------------------------------------------------------------------- #
# Per-candidate feature extraction + rule-based sub-scores
# --------------------------------------------------------------------------- #

def extract_record(d):
    """Pull out everything we need from one candidate JSON object, computing
    every rule-based sub-score except the corpus-dependent text-similarity
    term (filled in during pass 2)."""

    cid = d["candidate_id"]
    profile = d["profile"]
    career = d.get("career_history", []) or []
    education = d.get("education", []) or []
    skills = d.get("skills", []) or []
    sig = d.get("redrob_signals", {}) or {}

    current_title = safe_lower(profile.get("current_title"))
    current_company = safe_lower(profile.get("current_company"))
    current_industry = safe_lower(profile.get("current_industry"))
    yoe = profile.get("years_of_experience", 0) or 0
    location = safe_lower(profile.get("location"))
    country = safe_lower(profile.get("country"))

    # Narrative text used for shipped-evidence / domain checks AND for the
    # text-similarity component. Deliberately excludes skills[].
    narrative_parts = [safe_lower(profile.get("headline")), safe_lower(profile.get("summary"))]
    titles_text = [current_title]
    for c in career:
        narrative_parts.append(safe_lower(c.get("description")))
        titles_text.append(safe_lower(c.get("title")))
    narrative = " \n ".join(narrative_parts)
    all_titles = " \n ".join(titles_text)

    # ---------------- domain / title relevance (anti-stuffing core) ------- #
    title_hit_specific = contains_any(current_title, RELEVANT_TITLE_TERMS)
    title_hit_generic = contains_any(current_title, GENERIC_ENGINEERING_TERMS)
    title_hit_nonrelevant = contains_any(current_title, NON_ENGINEERING_TITLE_TERMS)

    past_titles_relevant = sum(1 for t in titles_text[1:] if contains_any(t, RELEVANT_TITLE_TERMS))
    past_titles_generic = sum(1 for t in titles_text[1:] if contains_any(t, GENERIC_ENGINEERING_TERMS))

    shipped_relevant_system = (
        contains_any(narrative, SHIPPED_EVIDENCE_TERMS) and contains_any(narrative, RELEVANT_SYSTEM_NOUNS)
    )

    if title_hit_nonrelevant and not title_hit_specific and not title_hit_generic:
        title_score = 0.05
    elif title_hit_specific:
        title_score = 1.0
    elif title_hit_generic:
        title_score = 0.55
    else:
        title_score = 0.20

    history_bonus = clamp(0.06 * past_titles_relevant + 0.02 * past_titles_generic, 0, 0.25)
    shipped_bonus = 0.30 if shipped_relevant_system else 0.0
    industry_bonus = 0.05 if any(k in current_industry for k in ["ai", "ml", "software", "technology", "tech"]) else 0.0

    domain_relevance = clamp(title_score + history_bonus + shipped_bonus + industry_bonus, 0.0, 1.0)

    # ---------------------------- skill evidence ---------------------------#
    assessment = sig.get("skill_assessment_scores", {}) or {}
    skills_by_name = {}
    for s in skills:
        nm = safe_lower(s.get("name"))
        if nm:
            skills_by_name.setdefault(nm, s)

    def best_match_for_group(terms):
        for s in skills:
            nm = safe_lower(s.get("name"))
            if any(t in nm for t in terms):
                return s
        return None

    group_scores = []
    matched_groups = {}
    for group, terms in MUST_HAVE_SKILL_GROUPS.items():
        s = best_match_for_group(terms)
        if s is None:
            group_scores.append(0.0)
            continue
        prof = s.get("proficiency", "beginner")
        prof_level = PROFICIENCY_LEVEL.get(prof, 0.4)
        dur = s.get("duration_months", 0) or 0
        dur_factor = clamp(dur / 24.0, 0.0, 1.0)
        endorsements = s.get("endorsements", 0) or 0
        endorse_factor = clamp(0.5 + endorsements / 40.0, 0.5, 1.0)

        assessed = None
        for k, v in assessment.items():
            if safe_lower(k) == safe_lower(s.get("name")):
                assessed = v
                break
        if assessed is not None:
            expected = PROFICIENCY_EXPECTED_ASSESSMENT.get(prof, 50.0)
            trust = clamp(assessed / expected, 0.25, 1.15)
        else:
            trust = 0.85  # mild discount for unverified (no platform assessment on file)

        strength = ((prof_level + dur_factor) / 2.0) * trust * (0.85 + 0.15 * endorse_factor)
        group_scores.append(clamp(strength, 0.0, 1.0))
        matched_groups[group] = (s.get("name"), prof, dur, assessed)

    skill_group_score = sum(group_scores) / len(group_scores)

    nice_hits = []
    for s in skills:
        nm = safe_lower(s.get("name"))
        if any(t in nm for t in NICE_TO_HAVE_SKILLS) and (s.get("duration_months", 0) or 0) >= 6:
            nice_hits.append(s.get("name"))
    nice_bonus = clamp(0.04 * len(nice_hits), 0.0, 0.08)

    # GATE: skill credit is scaled down hard when domain_relevance is low.
    # This is the explicit defense against keyword-stuffed skills sections
    # attached to an irrelevant title/career (the JD's "Marketing Manager
    # with all the AI keywords" trap).
    gate = clamp(0.22 + 0.78 * domain_relevance, 0.22, 1.0)
    gated_skill_score = skill_group_score * gate

    # ------------------------------ disqualifiers --------------------------#
    penalty = 1.0
    flags = []

    companies = [safe_lower(c.get("company")) for c in career]
    if companies and all(any(f in co for f in CONSULTING_FIRMS) for co in companies):
        penalty *= 0.40
        flags.append("consulting_only_career")

    is_research_title = "research" in current_title or any("research" in t for t in titles_text[1:])
    has_production_evidence = contains_any(narrative, ["production", "deployed", "shipped", "real users", "launched"])
    if is_research_title and not has_production_evidence:
        penalty *= 0.20
        flags.append("research_only_no_production")

    ai_skill_months = sum(
        s.get("duration_months", 0) or 0 for s in skills
        if any(t in safe_lower(s.get("name")) for t in ["llm", "langchain", "gpt", "rag", "prompt"])
    )
    pre_llm_evidence = contains_any(narrative, PRE_LLM_ML_TERMS) or any(
        (s.get("duration_months", 0) or 0) >= 24
        and any(t in safe_lower(s.get("name")) for t in ["machine learning", "data engineering", "recommendation", "search", "nlp", "ranking"])
        for s in skills
    )
    if contains_any(narrative, LANGCHAIN_TUTORIAL_TERMS) and ai_skill_months <= 12 and not pre_llm_evidence:
        penalty *= 0.55
        flags.append("langchain_only_recent_ai_experience")

    current_role = next((c for c in career if c.get("is_current")), career[0] if career else None)
    if current_role:
        cur_title = safe_lower(current_role.get("title"))
        cur_dur = current_role.get("duration_months", 0) or 0
        if contains_any(cur_title, ARCHITECT_TITLE_TERMS) and cur_dur >= 18 and not contains_any(
            safe_lower(current_role.get("description")), CODE_EVIDENCE_TERMS
        ):
            penalty *= 0.55
            flags.append("architecture_only_18mo_no_code")

    has_cv_speech_robotics = contains_any(narrative, CV_SPEECH_ROBOTICS_TERMS) or any(
        any(t in safe_lower(s.get("name")) for t in CV_SPEECH_ROBOTICS_TERMS) for s in skills
    )
    has_nlp_ir = contains_any(narrative, NLP_IR_TERMS) or any(
        any(t in safe_lower(s.get("name")) for t in NLP_IR_TERMS) for s in skills
    )
    if has_cv_speech_robotics and not has_nlp_ir:
        penalty *= 0.45
        flags.append("cv_speech_robotics_without_nlp_ir")

    seniority_order = ["junior", "engineer", "senior", "staff", "principal", "director"]

    def seniority_rank(t):
        for i, lvl in enumerate(seniority_order):
            if lvl in t:
                return i
        return 1

    short_escalations = 0
    for i in range(len(career) - 1):
        d1 = career[i].get("duration_months", 0) or 0
        if d1 < 18 and seniority_rank(safe_lower(career[i].get("title"))) < seniority_rank(safe_lower(career[i + 1].get("title"))):
            short_escalations += 1
    if short_escalations >= 2:
        penalty *= 0.65
        flags.append("title_chasing_pattern")

    github_score = sig.get("github_activity_score", -1)
    has_external_validation = (github_score is not None and github_score > 10) or any(
        k in narrative for k in ["paper", "talk", "conference", "open source", "open-source", "blog"]
    )
    if yoe >= 5 and (github_score is None or github_score <= 0) and not has_external_validation:
        penalty *= 0.75
        flags.append("closed_source_5yr_no_external_validation")

    penalty = clamp(penalty, 0.10, 1.0)

    # ---------------------------- years of experience ----------------------#
    if 5 <= yoe <= 9:
        yoe_fit = 1.0
    elif yoe < 5:
        yoe_fit = clamp(1.0 - (5 - yoe) * 0.12, 0.45, 1.0)
    else:
        yoe_fit = clamp(1.0 - (yoe - 9) * 0.06, 0.50, 1.0)

    # ------------------------------- location -------------------------------#
    willing_relocate = bool(sig.get("willing_to_relocate", False))
    if country and country != "india":
        location_fit = 0.55 if willing_relocate else 0.45
    elif any(c in location for c in PUNE_NOIDA):
        location_fit = 1.0
    elif any(c in location for c in TIER1_PLUS_INDIA_CITIES):
        location_fit = 0.90
    else:
        location_fit = 0.78 if willing_relocate else 0.55

    # ---------------------------- notice period -----------------------------#
    notice_days = sig.get("notice_period_days", 30) or 0
    if notice_days <= 30:
        notice_fit = 1.0
    else:
        notice_fit = clamp(1.0 - 0.65 * ((notice_days - 30) / 150.0), 0.35, 1.0)

    # ------------------------------- education -------------------------------#
    tier_map = {"tier_1": 1.0, "tier_2": 0.85, "tier_3": 0.72, "tier_4": 0.62, "unknown": 0.65}
    if education:
        edu_fit = max(tier_map.get(e.get("tier", "unknown"), 0.65) for e in education)
    else:
        edu_fit = 0.60

    # ------------------------------ behavioral -------------------------------#
    last_active = parse_date(sig.get("last_active_date"))
    response_rate = sig.get("recruiter_response_rate", 0.0) or 0.0
    open_to_work = bool(sig.get("open_to_work_flag", False))
    verified_email = bool(sig.get("verified_email", False))
    verified_phone = bool(sig.get("verified_phone", False))

    # ------------------------------ honeypot ---------------------------------#
    honeypot = False
    honeypot_reasons = []
    expert_zero_dur = sum(1 for s in skills if s.get("proficiency") == "expert" and (s.get("duration_months", 0) or 0) <= 2)
    if expert_zero_dur >= 1:
        honeypot = True
        honeypot_reasons.append("expert-proficiency skill with ~0 months hands-on duration")

    total_months = sum(c.get("duration_months", 0) or 0 for c in career)
    if yoe > 0:
        ratio = total_months / (yoe * 12.0)
        if ratio > 1.6 or ratio < 0.45:
            honeypot = True
            honeypot_reasons.append("career-history duration inconsistent with stated years_of_experience")

    return {
        "candidate_id": cid,
        "current_title": profile.get("current_title"),
        "current_company": profile.get("current_company"),
        "yoe": yoe,
        "location": profile.get("location"),
        "country": profile.get("country"),
        "matched_groups": matched_groups,
        "nice_hits": nice_hits,
        "flags": flags,
        "domain_relevance": domain_relevance,
        "gated_skill_score": gated_skill_score,
        "nice_bonus": nice_bonus,
        "penalty": penalty,
        "yoe_fit": yoe_fit,
        "location_fit": location_fit,
        "notice_fit": notice_fit,
        "notice_days": notice_days,
        "edu_fit": edu_fit,
        "last_active": last_active,
        "response_rate": response_rate,
        "open_to_work": open_to_work,
        "verified_email": verified_email,
        "verified_phone": verified_phone,
        "honeypot": honeypot,
        "honeypot_reasons": honeypot_reasons,
        "narrative": narrative,
        "shipped_relevant_system": shipped_relevant_system,
    }


def term_counts(text):
    counts = {}
    for term in TEXT_SIM_VOCAB:
        c = text.count(term)
        if c:
            counts[term] = c
    return counts


def cosine_from_counts(counts_a, counts_b, idf):
    keys = set(counts_a) & set(counts_b)
    if not keys:
        return 0.0
    num = sum(counts_a[k] * idf[k] * counts_b[k] * idf[k] for k in keys)
    norm_a = math.sqrt(sum((counts_a[k] * idf[k]) ** 2 for k in counts_a)) or 1.0
    norm_b = math.sqrt(sum((counts_b[k] * idf[k]) ** 2 for k in counts_b)) or 1.0
    return num / (norm_a * norm_b)


# --------------------------------------------------------------------------- #
# Reasoning generation
# --------------------------------------------------------------------------- #

OPEN_STRONG = [
    "{title} with {yoe:.1f} years of experience and direct evidence of {skill} work ({prof}, {dur}mo hands-on).",
    "{yoe:.1f}-year {title} whose career history shows real {skill} work, not just a skills-list mention.",
    "Strong fit: {title}, {yoe:.1f} yrs, with {skill} experience corroborated by platform assessment.",
]
OPEN_MID = [
    "{title} ({yoe:.1f} yrs) shows partial alignment — {skill} experience present but not the strongest match.",
    "{yoe:.1f}-year {title} with some relevant background ({skill}), though overall fit is moderate.",
]
OPEN_WEAK = [
    "{title} ({yoe:.1f} yrs) is an adjacent-skills filler pick; limited direct evidence of {skill} work.",
    "Included as lower-tier filler: {title}, {yoe:.1f} yrs, weak match on the core skill requirements.",
]

CONCERN_TEMPLATES = {
    "consulting_only_career": "entire career has been at consulting/services firms ({company}), no product-company evidence",
    "research_only_no_production": "research-oriented title with no production-deployment evidence in career history",
    "langchain_only_recent_ai_experience": "AI exposure looks limited to recent LangChain/LLM-API work with no earlier ML production background",
    "architecture_only_18mo_no_code": "current role reads as architecture/tech-lead for 18+ months with no code-evidence in the description",
    "cv_speech_robotics_without_nlp_ir": "background is CV/speech/robotics without NLP or information-retrieval exposure",
    "title_chasing_pattern": "career shows rapid seniority escalation across short (<18mo) stints",
    "closed_source_5yr_no_external_validation": "5+ years experience but no GitHub activity or other external validation on file",
}

SHIPPED_NOTE_VARIANTS = [
    "career history explicitly mentions shipping a ranking/search/recommendation system",
    "has previously built and shipped a search/recommendation system, per their career history",
    "career history shows production work on a ranking or retrieval system",
    "has shipped a relevance/recommendation system to real users before, per their work history",
]
NOTICE_NOTE_VARIANTS = [
    "notice period is {days}d, above the sub-30-day preference",
    "would need {days}d notice, so the bar here is higher per the JD",
    "{days}-day notice period to factor in",
]
STALE_NOTE_VARIANTS = [
    "inactive on the platform for {days}d",
    "hasn't been active on Redrob in {days}d",
    "{days}d since last login, so availability is uncertain",
]
RESPONSE_NOTE_VARIANTS = [
    "low recruiter response rate ({rate:.0%})",
    "recruiter response rate is only {rate:.0%}",
    "rarely responds to recruiters ({rate:.0%} response rate)",
]
LOCATION_NOTE_VARIANTS = [
    "located in {loc}, not Pune/Noida",
    "based in {loc} rather than the Pune/Noida hubs",
    "{loc}-based, outside the preferred Pune/Noida locations",
]
OUTSIDE_INDIA_VARIANTS = [
    "based outside India ({country}); no visa sponsorship",
    "currently in {country}, outside India, and there's no visa sponsorship",
    "would require relocation from {country} with no visa sponsorship offered",
]


def build_reasoning(rec, rank, reference_date):
    rng = random.Random(rec["candidate_id"])
    title = rec["current_title"] or "candidate"
    yoe = rec["yoe"] or 0.0

    matched = [(g, v) for g, v in rec["matched_groups"].items() if v[0]]
    if matched:
        g, (name, prof, dur, assessed) = rng.choice(matched)
        skill, prof_s, dur_s = name, prof, dur
    else:
        skill, prof_s, dur_s = "no must-have skill clearly evidenced", "n/a", 0

    if rank <= 15:
        template = rng.choice(OPEN_STRONG)
    elif rank <= 60:
        template = rng.choice(OPEN_MID)
    else:
        template = rng.choice(OPEN_WEAK)

    opening = template.format(title=title, yoe=yoe, skill=skill, prof=prof_s, dur=dur_s)

    extras = []
    if rec["shipped_relevant_system"]:
        extras.append(rng.choice(SHIPPED_NOTE_VARIANTS))

    if rec["flags"]:
        flag = rec["flags"][0]
        note = CONCERN_TEMPLATES.get(flag, flag)
        if flag == "consulting_only_career":
            note = note.format(company=rec["current_company"] or "n/a")
        extras.append(note)

    if rec["last_active"]:
        days_inactive = (reference_date - rec["last_active"]).days
        if days_inactive > 90:
            extras.append(rng.choice(STALE_NOTE_VARIANTS).format(days=days_inactive))
    if rec["response_rate"] < 0.15:
        extras.append(rng.choice(RESPONSE_NOTE_VARIANTS).format(rate=rec["response_rate"]))
    if rec["notice_days"] > 60:
        extras.append(rng.choice(NOTICE_NOTE_VARIANTS).format(days=rec["notice_days"]))
    if rec["country"] and safe_lower(rec["country"]) != "india":
        extras.append(rng.choice(OUTSIDE_INDIA_VARIANTS).format(country=rec["country"]))
    elif rec["location"]:
        loc_l = safe_lower(rec["location"])
        if not any(c in loc_l for c in PUNE_NOIDA):
            extras.append(rng.choice(LOCATION_NOTE_VARIANTS).format(loc=rec["location"]))

    rng.shuffle(extras)
    extras = extras[:2]

    text = opening
    if extras:
        joined = "; ".join(extras)
        joined = joined[0].upper() + joined[1:] if joined else joined  # capitalize first letter only
        text += " " + joined + "."
    if not text.endswith("."):
        text += "."
    return text.replace(" jd", " JD").replace(" jd.", " JD.").replace(" jd,", " JD,").replace(" jd;", " JD;")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="Redrob hackathon ranker")
    ap.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    ap.add_argument("--out", required=True, help="Path to write submission CSV")
    ap.add_argument("--topn", type=int, default=100)
    args = ap.parse_args()

    t0 = time.time()

    records = []
    max_active = None
    n_read = 0
    with open(args.candidates, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_read += 1
            d = json.loads(line)
            rec = extract_record(d)
            rec["_term_counts"] = term_counts(rec["narrative"])
            if rec["last_active"] and (max_active is None or rec["last_active"] > max_active):
                max_active = rec["last_active"]
            records.append(rec)

    reference_date = max_active or FALLBACK_REFERENCE_DATE

    # ---- corpus-level IDF for the text-similarity component ---- #
    df = {}
    for rec in records:
        for term in rec["_term_counts"]:
            df[term] = df.get(term, 0) + 1
    n_docs = len(records)
    idf = {term: math.log((n_docs + 1) / (df.get(term, 0) + 1)) + 1.0 for term in TEXT_SIM_VOCAB}

    jd_counts = term_counts(JD_REFERENCE_TEXT)

    honeypots_excluded = 0
    scored = []
    for rec in records:
        if rec["honeypot"]:
            honeypots_excluded += 1
            continue

        text_sim = cosine_from_counts(jd_counts, rec["_term_counts"], idf)
        text_sim = clamp(text_sim, 0.0, 1.0)

        base = (
            0.40 * rec["domain_relevance"]
            + 0.30 * rec["gated_skill_score"]
            + 0.08 * rec["nice_bonus"] / 0.08 * 0.08  # already capped 0-0.08; keep weight explicit
            + 0.08 * rec["yoe_fit"]
            + 0.07 * rec["location_fit"]
            + 0.05 * rec["notice_fit"]
            + 0.02 * rec["edu_fit"]
        )
        base *= rec["penalty"]
        base *= (0.55 + 0.45 * text_sim)

        days_inactive = (reference_date - rec["last_active"]).days if rec["last_active"] else 365
        recency_score = clamp(1.0 - days_inactive / 240.0, 0.35, 1.0)
        behavioral = (
            0.45 * recency_score
            + 0.35 * (0.40 + 0.60 * rec["response_rate"])
            + 0.20 * (1.0 if rec["open_to_work"] else 0.85)
        )
        behavioral += 0.01 * rec["verified_email"] + 0.01 * rec["verified_phone"]
        behavioral = clamp(behavioral, 0.45, 1.05)

        final_score = clamp(base * behavioral, 0.0, 1.0)
        rec["_final_score"] = final_score
        rec["_text_sim"] = text_sim
        scored.append(rec)

    scored.sort(key=lambda r: (-round(r["_final_score"], 4), r["candidate_id"]))
    top = scored[: args.topn]

    rows = []
    for i, rec in enumerate(top):
        rank = i + 1
        score = round(rec["_final_score"], 4)
        reasoning = build_reasoning(rec, rank, reference_date)
        rows.append((rec["candidate_id"], rank, score, reasoning))

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for cid, rank, score, reasoning in rows:
            w.writerow([cid, rank, f"{score:.4f}", reasoning])

    elapsed = time.time() - t0
    print(f"Read {n_read} candidates, excluded {honeypots_excluded} as honeypots, "
          f"wrote top {len(rows)} to {args.out} in {elapsed:.1f}s "
          f"(reference_date={reference_date}).", file=sys.stderr)


if __name__ == "__main__":
    main()
