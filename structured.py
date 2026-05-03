"""
Structured LLM Outputs — Core Extractor
=========================================
Get reliable, validated JSON from any LLM.
Zero manual JSON parsing. Zero brittle regex.

The Problem (and why this matters):
    "Prompt-only JSON extraction fails 5–20% of the time in production.
     At 10,000 API calls/day, 10% failure = 1,000 broken downstream processes."

This library fixes that with a 4-layer reliability stack:

    Layer 1 (Fast):   Native structured output mode (OpenAI/Anthropic)
    Layer 2 (Retry):  Instructor auto-retry with validation feedback
    Layer 3 (Repair): Schema-aware JSON repair for malformed responses
    Layer 4 (Parse):  Fuzzy extraction from plain text as last resort

Usage:
    from extractors.structured import StructuredExtractor, ExtractionMode

    # Simple — define schema, call extract
    class JobPosting(BaseModel):
        title: str
        company: str
        salary_min: Optional[float]
        salary_max: Optional[float]
        location: str
        is_remote: bool

    extractor = StructuredExtractor()
    result = extractor.extract(
        schema=JobPosting,
        text="Senior AI Engineer at Grab Singapore. RM 12,000–18,000/month. Hybrid.",
    )
    print(result.data)
    # JobPosting(title='Senior AI Engineer', company='Grab', ...)

    # Batch extraction
    results = extractor.extract_batch(schema=JobPosting, texts=[...])
"""

import json
import re
import time
import logging
from enum import Enum
from typing import Type, TypeVar, Optional, Any
from dataclasses import dataclass, field

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class ExtractionMode(str, Enum):
    INSTRUCTOR = "instructor"      # Instructor library (best, multi-provider)
    NATIVE = "native"              # Provider native structured output
    JSON_MODE = "json_mode"        # JSON mode (provider's built-in)
    PROMPT_ONLY = "prompt_only"    # Prompt engineering only (baseline)
    AUTO = "auto"                  # Auto-select best available


class ExtractionStatus(str, Enum):
    SUCCESS = "success"
    RETRY_SUCCESS = "retry_success"       # Succeeded after retries
    REPAIR_SUCCESS = "repair_success"     # Succeeded after JSON repair
    FUZZY_SUCCESS = "fuzzy_success"       # Succeeded with fuzzy extraction
    FAILED = "failed"


@dataclass
class ExtractionResult:
    """Result from a structured extraction call."""
    status: ExtractionStatus
    data: Optional[BaseModel]               # Validated Pydantic model
    raw_response: str                       # Raw LLM response
    attempts: int                           # Number of attempts made
    latency_ms: float
    mode_used: ExtractionMode
    errors: list[str] = field(default_factory=list)
    repair_applied: bool = False

    @property
    def success(self) -> bool:
        return self.status != ExtractionStatus.FAILED

    @property
    def data_dict(self) -> dict:
        return self.data.model_dump() if self.data else {}


# ── JSON repair utilities ─────────────────────────────────────────────────────

def _strip_markdown_json(text: str) -> str:
    """Remove markdown code fences from JSON output."""
    text = re.sub(r"^```(?:json)?\s*\n?", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"\n?```\s*$", "", text.strip(), flags=re.MULTILINE)
    return text.strip()


def _fix_trailing_commas(text: str) -> str:
    """Remove trailing commas before } or ]."""
    return re.sub(r",\s*([}\]])", r"\1", text)


def _fix_single_quotes(text: str) -> str:
    """Replace single quotes with double quotes in JSON."""
    # Basic replacement — handles simple cases
    text = re.sub(r"(?<![\\])'", '"', text)
    return text


def _fix_unquoted_keys(text: str) -> str:
    """Quote unquoted JSON keys."""
    return re.sub(r'(\{|\,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1 "\2":', text)


def repair_json(text: str) -> Optional[str]:
    """
    Attempt to repair malformed JSON using a sequence of fixes.
    Returns fixed JSON string or None if unrecoverable.
    """
    # Step 1: Strip markdown fences
    text = _strip_markdown_json(text)

    # Step 2: Try direct parse
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    # Step 3: Apply progressive fixes
    fixes = [
        _fix_trailing_commas,
        _fix_single_quotes,
        _fix_unquoted_keys,
    ]

    current = text
    for fix in fixes:
        current = fix(current)
        try:
            json.loads(current)
            return current
        except json.JSONDecodeError:
            continue

    # Step 4: Extract JSON object from text
    json_match = re.search(r"\{[\s\S]+\}", text, re.DOTALL)
    if json_match:
        candidate = json_match.group()
        candidate = _fix_trailing_commas(candidate)
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

    return None


def fuzzy_extract(schema: Type[T], text: str) -> Optional[T]:
    """
    Last-resort extraction: scan text for field values using regex.
    Only works for simple schemas with basic field types.
    """
    fields = schema.model_fields
    data = {}

    for field_name, field_info in fields.items():
        # Try to find the field value in text
        pattern = re.compile(
            rf'(?:"{field_name}"|{field_name})\s*[:\s]+([^,\n\]}}]+)',
            re.IGNORECASE
        )
        match = pattern.search(text)
        if match:
            raw_value = match.group(1).strip().strip('"\'')

            # Type coercion
            annotation = field_info.annotation
            try:
                if annotation in (int, Optional[int]):
                    data[field_name] = int(re.sub(r"[^\d]", "", raw_value))
                elif annotation in (float, Optional[float]):
                    data[field_name] = float(re.sub(r"[^\d.]", "", raw_value))
                elif annotation in (bool, Optional[bool]):
                    data[field_name] = raw_value.lower() in ("true", "yes", "1")
                else:
                    data[field_name] = raw_value
            except (ValueError, TypeError):
                pass

    if not data:
        return None

    try:
        return schema(**data)
    except ValidationError:
        return None


# ── Main extractor ────────────────────────────────────────────────────────────

class StructuredExtractor:
    """
    Multi-provider structured LLM output extractor.

    Automatically selects the best available extraction method
    and falls back through layers until it gets valid output.

    Supported providers:
    - OpenAI (GPT-4o, GPT-4o-mini, o1)
    - Anthropic (Claude Opus, Sonnet, Haiku)
    - Ollama (local — any model, free)
    - Any OpenAI-compatible endpoint

    Usage:
        extractor = StructuredExtractor(provider="ollama", model="qwen2.5:7b")
        result = extractor.extract(MySchema, text="...")
    """

    def __init__(
        self,
        provider: str = "ollama",       # "openai" | "anthropic" | "ollama"
        model: str = "qwen2.5:7b",      # Model name
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,  # Custom endpoint (Ollama default: http://localhost:11434)
        max_retries: int = 3,
        mode: ExtractionMode = ExtractionMode.AUTO,
        temperature: float = 0.0,        # 0 = most deterministic
        enable_repair: bool = True,
        enable_fuzzy: bool = True,
    ):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.max_retries = max_retries
        self.mode = mode
        self.temperature = temperature
        self.enable_repair = enable_repair
        self.enable_fuzzy = enable_fuzzy
        self._client = None
        self._instructor_client = None

    def _get_client(self):
        """Lazy-initialise the LLM client."""
        if self._client is not None:
            return self._client

        if self.provider == "openai":
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )

        elif self.provider == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)

        elif self.provider == "ollama":
            from openai import OpenAI
            self._client = OpenAI(
                api_key="ollama",   # Dummy key for Ollama
                base_url=self.base_url or "http://localhost:11434/v1",
            )

        else:
            raise ValueError(f"Unknown provider: {self.provider}")

        return self._client

    def _get_instructor_client(self):
        """Get Instructor-wrapped client."""
        if self._instructor_client is not None:
            return self._instructor_client

        try:
            import instructor

            client = self._get_client()

            if self.provider in ("openai", "ollama"):
                self._instructor_client = instructor.from_openai(client)
            elif self.provider == "anthropic":
                self._instructor_client = instructor.from_anthropic(client)

            return self._instructor_client

        except ImportError:
            logger.warning("Instructor not installed. Run: pip install instructor")
            return None

    def _build_prompt(self, schema: Type[BaseModel], text: str) -> str:
        """Build extraction prompt with schema description."""
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        return (
            f"Extract the following information from the text and return it "
            f"as a JSON object matching this exact schema:\n\n"
            f"Schema:\n{schema_json}\n\n"
            f"Text to extract from:\n{text}\n\n"
            f"Return ONLY the JSON object. No explanation, no markdown fences, "
            f"no preamble. Just the raw JSON."
        )

    def extract(
        self,
        schema: Type[T],
        text: str,
        context: Optional[str] = None,
    ) -> ExtractionResult:
        """
        Extract structured data from text according to the given schema.

        Tries extraction methods in order:
        1. Instructor (with auto-retry on validation failure)
        2. JSON repair (if response is malformed JSON)
        3. Fuzzy extraction (last resort regex-based)

        Args:
            schema:  Pydantic model class defining the output structure
            text:    Input text to extract from
            context: Optional system context / domain description

        Returns:
            ExtractionResult with validated data
        """
        t0 = time.time()
        errors = []
        attempts = 0

        prompt = self._build_prompt(schema, text)
        system = context or f"You are a data extraction assistant. Extract structured information accurately."

        # ── Attempt 1: Instructor ──────────────────────────────────────────
        instructor_client = self._get_instructor_client()
        if instructor_client is not None:
            try:
                data = instructor_client.chat.completions.create(
                    model=self.model,
                    response_model=schema,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    max_retries=self.max_retries,
                    temperature=self.temperature,
                )
                latency = (time.time() - t0) * 1000
                return ExtractionResult(
                    status=ExtractionStatus.SUCCESS,
                    data=data,
                    raw_response=str(data),
                    attempts=1,
                    latency_ms=round(latency, 1),
                    mode_used=ExtractionMode.INSTRUCTOR,
                )
            except Exception as e:
                errors.append(f"Instructor: {str(e)[:100]}")
                attempts += 1

        # ── Attempt 2: Raw JSON mode with repair ──────────────────────────
        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system + "\nAlways respond with valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
            )
            raw = response.choices[0].message.content or ""
            attempts += 1

            # Try direct parse
            try:
                parsed = json.loads(raw)
                data = schema(**parsed)
                latency = (time.time() - t0) * 1000
                return ExtractionResult(
                    status=ExtractionStatus.SUCCESS,
                    data=data,
                    raw_response=raw,
                    attempts=attempts,
                    latency_ms=round(latency, 1),
                    mode_used=ExtractionMode.JSON_MODE,
                )
            except (json.JSONDecodeError, ValidationError) as e:
                errors.append(f"Direct parse: {str(e)[:100]}")

            # Try repair
            if self.enable_repair:
                repaired = repair_json(raw)
                if repaired:
                    try:
                        parsed = json.loads(repaired)
                        data = schema(**parsed)
                        latency = (time.time() - t0) * 1000
                        return ExtractionResult(
                            status=ExtractionStatus.REPAIR_SUCCESS,
                            data=data,
                            raw_response=raw,
                            attempts=attempts,
                            latency_ms=round(latency, 1),
                            mode_used=ExtractionMode.JSON_MODE,
                            repair_applied=True,
                        )
                    except (json.JSONDecodeError, ValidationError) as e:
                        errors.append(f"Repair: {str(e)[:100]}")

            # Try fuzzy extraction
            if self.enable_fuzzy:
                fuzzy_result = fuzzy_extract(schema, raw)
                if fuzzy_result:
                    latency = (time.time() - t0) * 1000
                    return ExtractionResult(
                        status=ExtractionStatus.FUZZY_SUCCESS,
                        data=fuzzy_result,
                        raw_response=raw,
                        attempts=attempts,
                        latency_ms=round(latency, 1),
                        mode_used=ExtractionMode.PROMPT_ONLY,
                    )

        except Exception as e:
            errors.append(f"Raw call: {str(e)[:100]}")

        latency = (time.time() - t0) * 1000
        return ExtractionResult(
            status=ExtractionStatus.FAILED,
            data=None,
            raw_response="",
            attempts=attempts,
            latency_ms=round(latency, 1),
            mode_used=ExtractionMode.PROMPT_ONLY,
            errors=errors,
        )

    def extract_batch(
        self,
        schema: Type[T],
        texts: list[str],
        context: Optional[str] = None,
    ) -> list[ExtractionResult]:
        """Extract from multiple texts."""
        return [self.extract(schema, text, context) for text in texts]
