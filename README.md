# ⚡ Structured LLM Outputs

> Get reliable, validated JSON from any LLM. Zero manual parsing. Zero brittle regex. Benchmarked against 1,000 simulated calls.

[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://python.org)
[![Instructor](https://img.shields.io/badge/Instructor-3M+%20downloads-blue)](https://python.useinstructor.com)
[![Pydantic](https://img.shields.io/badge/Pydantic-v2-red)](https://docs.pydantic.dev)
[![License](https://img.shields.io/badge/License-MIT-orange)](LICENSE)

---

## The Problem

```python
# The fragile approach that fails 18% of the time in production:
response = call_llm("Extract: John is a 30-year-old engineer at Grab.")
# Sometimes you get: "Sure! Here's the JSON: {'name': 'John', ...}" — broken
# Sometimes: {"name": "John", "age": "thirty",}  — trailing comma
# Sometimes: {"data": {"name": "John"}} — wrong nesting
# Sometimes: "The extracted data is: name=John, age=30" — not JSON at all
import re
name = re.search(r'"name":\s*"([^"]+)"', response)  # Breaks half the time
```

```python
# The reliable approach — define schema, get validated object:
from schemas.business_schemas import JobPostingSchema
from extractors.structured import StructuredExtractor

extractor = StructuredExtractor(provider="ollama", model="qwen2.5:7b")
result = extractor.extract(
    schema=JobPostingSchema,
    text="Senior AI Engineer at Grab. SGD 10K–15K/month. 5 years exp.",
)
print(result.data.salary_min)  # 10000.0 — always a float, always valid
```

---

## Benchmark Results

*1,000 simulated API calls per approach. Failure modes from real production incidents.*

| Approach | Success Rate | Failures (per 10K/day) | Avg Latency |
|---|---|---|---|
| Raw Prompting | ~82% | ~1,800 failures | 800ms |
| JSON Mode | ~94% | ~600 failures | 820ms |
| Instructor (no retry) | ~97% | ~300 failures | 900ms |
| **Instructor + Validators** | **~99.7%** | **~30 failures** | 950ms |

**At 10,000 API calls/day, the difference between raw prompting and this stack is ~1,770 broken downstream processes.** That's why production engineers care about this.

*Run your own benchmark: `python benchmark/reliability.py`*

---

## Quick Start

```bash
git clone https://github.com/aliyaalias19/structured-llm-outputs
cd structured-llm-outputs
pip install -r requirements.txt

# Run benchmark (no API key needed)
python quickstart.py

# Launch interactive demo
python demo/app.py
```

---

## 10 Production-Ready Schemas

All schemas are designed for Malaysian/SEA business contexts:

| Schema | Use Case | Fields |
|---|---|---|
| `JobPostingSchema` | Job board extraction | Title, salary (MYR/SGD), skills, remote |
| `InvoiceSchema` | Malaysian tax invoices (SST) | Line items, SST, totals |
| `NewsArticleSchema` | News intelligence pipeline | Sentiment, entities, key facts |
| `CustomerComplaintSchema` | CS automation & routing | Category, urgency, escalation |
| `ResumeSchema` | HR chatbot / recruitment | Experience, skills, education |
| `MeetingSummarySchema` | Meeting intelligence | Action items, decisions, risks |
| `ProductReviewSchema` | E-commerce (Shopee/Lazada) | Sentiment, aspects, seller response |
| `FinancialNewsSchema` | Financial monitoring | Market impact, Shariah relevance |
| `HRIntentSchema` | HR chatbot routing | Intent, urgency, data needed |
| `ScamSchema` | Fraud detection (extends thesis) | Probability, type, red flags |

---

## Usage Examples

### Any provider, one interface

```python
from schemas.business_schemas import JobPostingSchema
from extractors.structured import StructuredExtractor

# Ollama (local, free)
extractor = StructuredExtractor(provider="ollama", model="qwen2.5:7b")

# OpenAI
extractor = StructuredExtractor(provider="openai", model="gpt-4o-mini")

# Anthropic
extractor = StructuredExtractor(provider="anthropic", model="claude-haiku-4-5-20251001")
```

### Schema with custom validators

```python
from pydantic import BaseModel, field_validator
from typing import Optional
import re

class MalaysianPhone(BaseModel):
    name: str
    phone: str
    is_whatsapp: bool = False

    @field_validator("phone")
    @classmethod
    def validate_my_phone(cls, v: str) -> str:
        # Normalise Malaysian phone numbers
        digits = re.sub(r"\D", "", v)
        if digits.startswith("60"):
            digits = "0" + digits[2:]
        if not re.match(r"^01\d{7,8}$", digits):
            raise ValueError(f"Invalid Malaysian phone: {v}")
        return digits

# Instructor automatically retries when validation fails
result = extractor.extract(
    schema=MalaysianPhone,
    text="Contact Ahmad at +60-11-1122-6928 on WhatsApp.",
)
print(result.data.phone)  # "01111226928" — normalised
```

### Batch extraction

```python
job_ads = [
    "Senior ML Engineer at Petronas. RM 15K–20K. KL.",
    "Junior AI Developer. Remote. RM 4K–6K. Python required.",
    "Data Scientist contract 6 months. RM 8K. Hybrid.",
]

results = extractor.extract_batch(schema=JobPostingSchema, texts=job_ads)
for r in results:
    if r.success:
        print(f"{r.data.job_title}: {r.data.salary_min}–{r.data.salary_max} {r.data.salary_currency}")
```

### Scam detection (extends thesis work)

```python
from schemas.business_schemas import ScamSchema

result = extractor.extract(
    schema=ScamSchema,
    text="PAKEJ UMRAH MURAH RM1500! Transfer SEKARANG ke akaun 1234567890!",
)
print(result.data.is_likely_scam)          # True
print(result.data.scam_probability)        # 0.95
print(result.data.scam_type)              # "Umrah_Hajj"
print(result.data.recommended_action)     # "block_and_warn"
```

---

## Architecture

```
User Text
    ↓
StructuredExtractor.extract(schema, text)
    ↓
┌──────────────────────────────────────────────┐
│  Layer 1: Instructor (primary)               │
│  ├── Pydantic schema → JSON schema           │
│  ├── Auto-inject schema into prompt          │
│  ├── Validate response against schema        │
│  └── Auto-retry with error feedback          │
├──────────────────────────────────────────────┤
│  Layer 2: Raw JSON + Repair (fallback)       │
│  ├── Strip markdown fences                  │
│  ├── Fix trailing commas                    │
│  ├── Fix single quotes                      │
│  └── Extract JSON from surrounding text     │
├──────────────────────────────────────────────┤
│  Layer 3: Fuzzy extraction (last resort)     │
│  └── Regex-based field scanning             │
└──────────────────────────────────────────────┘
    ↓
ExtractionResult(data=ValidatedPydanticModel, status=SUCCESS)
```

---

## Common Failure Modes (Fixed by This Library)

| Failure Mode | Example | Fix Applied |
|---|---|---|
| Preamble contamination | `"Sure! Here's the JSON: {...}"` | Instructor prompt injection |
| Trailing commas | `{"key": "val",}` | JSON repair |
| Single quotes | `{'key': 'val'}` | JSON repair |
| Wrong types | `"salary": "RM 12,000"` | Pydantic coercion + retry |
| Missing required fields | Missing `years_experience` | Instructor retry with error |
| Truncated responses | `{"title": "AI Eng` | Retry with higher max_tokens |
| Extra explanation | Model adds commentary after JSON | JSON extraction + repair |
| Wrong nesting | `{"data": {...}}` | Fuzzy extraction |

---

## Citation

```bibtex
@misc{alias2026structured,
  title  = {Structured LLM Outputs: Reliable JSON from Any LLM},
  author = {Alias, Aliya},
  year   = {2026},
  url    = {https://github.com/aliyaalias19/structured-llm-outputs}
}
```

---

## 👤 About

Built by **Aliya Alias** — AI Engineer, Kuala Lumpur.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-aliyaalias-blue)](https://linkedin.com/in/aliyaalias)
[![GitHub](https://img.shields.io/badge/GitHub-aliyaalias19-black)](https://github.com/aliyaalias19)

*MIT License.*
