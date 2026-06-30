# Redrob Hackathon — Intelligent Candidate Discovery & Ranking Challenge

Team **K3-SpiritCourage**. Ranks the 100,000-candidate pool against the
"Senior AI Engineer — Founding Team" job description and produces the
top-100 submission CSV required by `submission_spec.md`.

## Quick start

```bash
pip install -r requirements.txt   # only needed for the optional xlsx converter
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

That single command is the full ranking step. It is deterministic, reads
`candidates.jsonl` once, and writes `submission.csv` in the required
`candidate_id,rank,score,reasoning` format.

Measured on a 4-core CPU-only Linux sandbox (Ubuntu 22.04, Python 3.10.12,
no GPU, no network): **~18 seconds wall-clock, ~660 MB peak RAM** for the
full 100,000-row pool — well inside the 5-minute / 16 GB / CPU-only / no-network
budget in `submission_spec.md` Section 3. No pre-computation step is required.

To produce the `.xlsx` copy needed for the hack2skill upload form:

```bash
python csv_to_xlsx.py --csv ./submission.csv --out ./submission.xlsx
```

## Repository contents

| File | Purpose |
|---|---|
| `rank.py` | The full ranking step. Zero third-party dependencies. |
| `csv_to_xlsx.py` | Converts the CSV to `.xlsx` for portal upload (not part of the scored ranking step). |
| `submission.csv` | The CSV this code produces, committed for review convenience. |
| `requirements.txt` | Only needed for `csv_to_xlsx.py` (`openpyxl`). |
| `submission_metadata.yaml` | Portal metadata, mirroring what was submitted on hack2skill. |

`candidates.jsonl` is **not** committed (487 MB, and provided by the
organizers) — supply your own copy via `--candidates`.

## Methodology

We treated the JD's own framing — "the right answer is not finding
candidates whose skills section contains the most AI keywords" — as the
central design constraint, not a footnote. Concretely:

1. **Title/career substance gates skill credit.** Every candidate gets a
   `domain_relevance` score from their current title, past titles, and
   whether their career-history descriptions show evidence of actually
   shipping a ranking/search/recommendation system in production. Skill-list
   evidence is then *multiplied* by this gate (floor ~0.22x for an
   irrelevant title/career, full credit for a clearly relevant one). This is
   the direct defense against the "Marketing Manager with every AI keyword"
   trap the JD describes.
2. **Skill claims are cross-checked, not taken at face value.** For each of
   the JD's four "absolutely need" skill areas (embeddings/retrieval, vector
   DB / hybrid search, Python, ranking-evaluation frameworks), we look for a
   matching skill entry and weight it by `duration_months`, `endorsements`,
   and — critically — the candidate's own `skill_assessment_scores` from
   `redrob_signals`. A self-rated "expert" with a low platform assessment
   score is discounted, not trusted. (Spot-checking the dataset surfaced
   exactly this pattern: a Backend Engineer with "advanced" self-rated NLP/
   Image Classification/Speech Recognition but assessment scores in the
   38–65 range — a corroboration mismatch our scoring catches.)
3. **JD disqualifiers are explicit penalty rules**, applied multiplicatively
   rather than as hard cutoffs: consulting-only careers (TCS/Infosys/Wipro/
   Accenture/Cognizant/Capgemini/etc. with zero product-company experience),
   research-only backgrounds with no production evidence, LangChain-only
   AI exposure under 12 months with no pre-LLM ML background, "architect/
   tech lead for 18+ months with no code in the role description," CV/
   speech/robotics backgrounds with no NLP/IR exposure, and rapid
   title-escalation across short stints.
4. **Years-of-experience, location, and notice period are soft, smooth
   functions**, never hard filters — per the JD's explicit "this is a range,
   not a requirement" framing. 5–9 years peaks; outside that band decays
   gently. Pune/Noida scores highest; other Tier-1 Indian cities score
   slightly lower; non-India is a moderate penalty (no visa sponsorship) but
   not exclusion. Notice period decays smoothly past 30 days instead of a
   cliff.
5. **A light, curated text-similarity score** (~10–15% effective weight)
   compares each candidate's career-history *descriptions* and profile
   summary — **never the skills[] list** — against a short reference passage
   built from the JD's "ideal candidate" paragraph, using a small hand-picked
   vocabulary (the same must-have/nice-to-have phrases plus "shipped",
   "production", "scale", etc.) with corpus-derived IDF weighting. Excluding
   the skills array here is deliberate: it's the same anti-keyword-stuffing
   defense as (1), applied to the text-similarity term specifically so it
   can't be gamed by a stuffed skills section either.
6. **Behavioral signals are a multiplicative availability modifier**
   (recency of `last_active_date`, `recruiter_response_rate`,
   `open_to_work_flag`, verified email/phone), bounded to roughly [0.45, 1.05]
   so they shape but don't dominate the ranking — directly implementing the
   JD's "a perfect-on-paper candidate who hasn't logged in for 6 months ...
   is, for hiring purposes, not actually available" guidance.
7. **Honeypot exclusion.** We explicitly detect and exclude two internally-
   inconsistent profile patterns before ranking: (a) "expert" proficiency
   claimed on a skill with ~0 months of hands-on duration, and (b) total
   career-history duration wildly inconsistent with the candidate's stated
   `years_of_experience`. On the full 100K pool this excludes 68 candidates;
   none reach the top 100, so the submission's honeypot rate is 0%.

Every reasoning string in `submission.csv` is generated from the same fields
used in scoring (title, years of experience, the specific matched skill +
proficiency + duration, any disqualifier flag, location, notice period,
activity recency) — nothing is invented, and phrasing is varied per
candidate (seeded by `candidate_id`) so the 100 reasonings are not
templated repeats of each other.

## Compute environment used for development/testing

- Platform: Linux sandbox (Ubuntu 22.04.5 LTS, aarch64), 4 CPU cores, ~3.8 GB RAM available
- Python 3.10.12
- No GPU used at any point; no network calls during the ranking step
- See `submission_metadata.yaml` for the full declaration

## AI tools used

Claude was used for: reading/summarizing the hackathon bundle docs, design
discussion on the scoring approach, writing and iterating on `rank.py`, and
spot-checking the dataset for trap patterns. No candidate data was sent to
any hosted LLM API as part of the ranking step itself — `rank.py` makes zero
network calls. See `submission_metadata.yaml` for the full declaration.
