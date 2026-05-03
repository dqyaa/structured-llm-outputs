"""
Quick Start — Structured LLM Outputs
=====================================
Run this first. No API key needed.

Usage:
    python quickstart.py
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

print("⚡ Structured LLM Outputs — Quick Start")
print("=" * 55)

# ── Step 1: Imports ───────────────────────────────────────────────────────────
print("\n📦 Checking imports...")
try:
    from pydantic import BaseModel, ValidationError
    from schemas.business_schemas import ALL_SCHEMAS, JobPostingSchema, ScamSchema
    from extractors.structured import repair_json, fuzzy_extract
    from benchmark.reliability import run_simulation, print_report, generate_markdown_report
    print(f"   ✅ All imports OK | {len(ALL_SCHEMAS)} schemas loaded")
except ImportError as e:
    print(f"   ❌ {e}")
    print("   Run: pip install -r requirements.txt")
    sys.exit(1)

# ── Step 2: Schema validation ─────────────────────────────────────────────────
print("\n📋 Testing Pydantic schemas...")

test_cases = [
    ("JobPostingSchema", JobPostingSchema, {
        "job_title": "Senior AI Engineer",
        "company_name": "GovTech Singapore",
        "location": "Singapore",
        "salary_min": 10000.0,
        "salary_max": 15000.0,
        "salary_currency": "SGD",
        "salary_period": "monthly",
        "is_remote": True,
        "required_skills": ["Python", "LangChain", "PyTorch"],
        "years_experience_min": 5,
    }),
    ("ScamSchema", ScamSchema, {
        "is_likely_scam": True,
        "scam_probability": 0.95,
        "scam_type": "Umrah_Hajj",
        "urgency_manipulation": True,
        "payment_requested": True,
        "red_flags": ["Urgent payment", "Unverified seller", "Too cheap"],
        "recommended_action": "block_and_warn",
        "explanation": "Classic Umrah scam with urgency manipulation and direct transfer request.",
    }),
]

for name, schema_class, data in test_cases:
    try:
        obj = schema_class(**data)
        d = obj.model_dump(exclude_none=True)
        print(f"   ✅ {name}: {len(d)} fields validated")
    except Exception as e:
        print(f"   ❌ {name}: {e}")

# ── Step 3: JSON repair ───────────────────────────────────────────────────────
print("\n🔧 Testing JSON repair...")

repair_tests = [
    (
        '{"title": "AI Engineer", "salary": 12000,}',
        "Trailing comma",
    ),
    (
        "```json\n{\"title\": \"ML Engineer\", \"salary\": 9000}\n```",
        "Markdown fences",
    ),
    (
        "Here's the JSON: {\"title\": \"Data Scientist\", \"salary\": 8000}",
        "Preamble text",
    ),
]

repair_pass = 0
for broken_json, label in repair_tests:
    repaired = repair_json(broken_json)
    ok = repaired is not None
    icon = "✅" if ok else "❌"
    print(f"   {icon} [{label}] repaired={ok}")
    if ok:
        repair_pass += 1

print(f"\n   JSON repair: {repair_pass}/{len(repair_tests)} fixed")

# ── Step 4: Reliability simulation ───────────────────────────────────────────
print("\n📊 Running reliability benchmark (1,000 simulated calls)...")
results = run_simulation(n_calls=1000)
print()
print_report(results, 1000)

# Save markdown report
md = generate_markdown_report(results, 1000)
Path("benchmark").mkdir(exist_ok=True)
Path("benchmark/results.md").write_text(md)
print("💾 Benchmark results saved to benchmark/results.md")
print("   ⬆️  Copy this into your README.md!\n")

# ── Step 5: Check optional dependencies ──────────────────────────────────────
print("🔌 Optional dependencies:")

try:
    import instructor
    print(f"   ✅ instructor (v{instructor.__version__})")
except ImportError:
    print("   ⚠️  instructor not installed (run: pip install instructor)")

try:
    import openai
    print("   ✅ openai SDK")
except ImportError:
    print("   ⚠️  openai not installed (needed for OpenAI/Ollama)")

try:
    import anthropic
    print("   ✅ anthropic SDK")
except ImportError:
    print("   ⚠️  anthropic not installed (needed for Claude)")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print("✅ Core pipeline working! No API key needed for benchmark.")
print("\nNext steps:")
print("  1. Run the benchmark:   python benchmark/reliability.py")
print("  2. Launch the demo:     python demo/app.py")
print("  3. Use Ollama locally:  ollama pull qwen2.5:7b")
print("  4. Test with real API:  set OPENAI_API_KEY= or ANTHROPIC_API_KEY=")
print(f"{'='*55}\n")
