"""
Production-Ready Schemas — Malaysian/SEA Business Contexts
===========================================================
10 schemas covering the most common structured extraction use cases
across Malaysian enterprise, fintech, and government applications.

Each schema includes:
- Appropriate field types and constraints
- Optional fields for graceful handling of missing data
- Field descriptions that guide the LLM
- Custom validators where needed
- Example usage in docstrings

Usage:
    from schemas.business_schemas import JobPostingSchema
    from extractors.structured import StructuredExtractor

    extractor = StructuredExtractor()
    result = extractor.extract(
        schema=JobPostingSchema,
        text="Senior AI Engineer at Grab. RM 12,000–18,000/month..."
    )
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, Literal
from enum import Enum
import re


# ── 1. Job Posting Extractor ──────────────────────────────────────────────────

class SeniorityLevel(str, Enum):
    INTERN = "intern"
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    LEAD = "lead"
    MANAGER = "manager"
    DIRECTOR = "director"
    VP = "vp"
    C_LEVEL = "c_level"


class JobPostingSchema(BaseModel):
    """
    Extract structured data from job postings.
    Works for Malaysian, Singaporean, and Australian job ads.

    Example input:
        "Senior AI Engineer at Grab Singapore. 5+ years experience.
         SGD 8,000–12,000/month. Hybrid work. LLM, Python required."
    """
    job_title: str = Field(description="Exact job title as posted")
    company_name: str = Field(description="Company or organisation name")
    location: str = Field(description="City and country e.g. 'Kuala Lumpur, Malaysia'")
    seniority: Optional[SeniorityLevel] = Field(None, description="Seniority level")
    salary_min: Optional[float] = Field(None, description="Minimum salary (numbers only)")
    salary_max: Optional[float] = Field(None, description="Maximum salary (numbers only)")
    salary_currency: Optional[str] = Field(None, description="Currency code: MYR, SGD, AUD, USD")
    salary_period: Optional[Literal["monthly", "annual"]] = Field(None)
    is_remote: Optional[bool] = Field(None, description="True if remote/WFH allowed")
    employment_type: Optional[Literal["full_time", "part_time", "contract", "internship"]] = None
    years_experience_min: Optional[int] = Field(None, ge=0, le=30)
    years_experience_max: Optional[int] = Field(None, ge=0, le=30)
    required_skills: list[str] = Field(default_factory=list, description="Required technical skills")
    preferred_skills: list[str] = Field(default_factory=list)
    education_required: Optional[str] = None
    visa_sponsorship: Optional[bool] = None
    application_deadline: Optional[str] = None

    @field_validator("salary_min", "salary_max", mode="before")
    @classmethod
    def clean_salary(cls, v):
        if v is None:
            return v
        if isinstance(v, str):
            v = re.sub(r"[^\d.]", "", v)
            return float(v) if v else None
        return v

    @field_validator("required_skills", "preferred_skills", mode="before")
    @classmethod
    def clean_skills(cls, v):
        if isinstance(v, str):
            return [s.strip() for s in re.split(r"[,;/]", v) if s.strip()]
        return v or []


# ── 2. Malaysian Invoice Parser ───────────────────────────────────────────────

class InvoiceLineItem(BaseModel):
    description: str
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    amount: float
    tax_code: Optional[str] = None


class InvoiceSchema(BaseModel):
    """
    Extract structured data from Malaysian tax invoices.
    Handles SST (Sales and Service Tax) format.

    Example input:
        "TAX INVOICE #INV-2024-001. Buyer: Aliya Tech Sdn Bhd.
         Consulting services: RM 5,000. SST 8%: RM 400. Total: RM 5,400."
    """
    invoice_number: str = Field(description="Invoice number / reference")
    invoice_date: Optional[str] = None
    due_date: Optional[str] = None
    seller_name: str = Field(description="Seller/vendor company name")
    seller_registration: Optional[str] = Field(None, description="SSM or company registration")
    buyer_name: Optional[str] = None
    buyer_ic_or_reg: Optional[str] = Field(None, description="Buyer IC or company reg number")
    line_items: list[InvoiceLineItem] = Field(default_factory=list)
    subtotal: Optional[float] = None
    tax_type: Optional[Literal["SST", "GST", "SERVICE_TAX", "NONE"]] = None
    tax_rate_percent: Optional[float] = None
    tax_amount: Optional[float] = None
    discount_amount: Optional[float] = None
    total_amount: float = Field(description="Final total amount due")
    currency: str = Field(default="MYR")
    payment_terms: Optional[str] = None

    @field_validator("total_amount", "subtotal", "tax_amount", mode="before")
    @classmethod
    def clean_amount(cls, v):
        if isinstance(v, str):
            v = re.sub(r"[^\d.]", "", v)
            return float(v) if v else None
        return v


# ── 3. News Article Intelligence ─────────────────────────────────────────────

class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class NewsArticleSchema(BaseModel):
    """
    Extract structured metadata and intelligence from news articles.
    Used for news monitoring pipelines (like Aliya's Aevoco work).

    Example input:
        "Maybank Q3 profit rises 12% to RM 2.4 billion, beating analyst forecasts.
         CEO Khairussaleh expects strong momentum to continue into 2025..."
    """
    headline: str = Field(description="Article headline or main title")
    summary: str = Field(max_length=500, description="2-3 sentence summary")
    sentiment: Sentiment = Field(description="Overall sentiment of the article")
    sentiment_score: float = Field(ge=-1.0, le=1.0, description="-1=very negative, 0=neutral, 1=very positive")
    topic_category: str = Field(description="e.g. Finance, Technology, Politics, Business")
    organisations_mentioned: list[str] = Field(default_factory=list)
    people_mentioned: list[str] = Field(default_factory=list)
    countries_mentioned: list[str] = Field(default_factory=list)
    key_facts: list[str] = Field(default_factory=list, description="Top 3-5 key facts as bullet points")
    financial_figures: list[str] = Field(default_factory=list, description="Any financial numbers mentioned")
    is_breaking_news: bool = False
    requires_follow_up: bool = Field(description="True if this article likely has follow-up stories")
    publication_date: Optional[str] = None
    source_name: Optional[str] = None


# ── 4. Customer Complaint Classifier ─────────────────────────────────────────

class UrgencyLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CustomerComplaintSchema(BaseModel):
    """
    Classify and extract data from customer complaints.
    Used for customer service automation systems.

    Example input:
        "I've been waiting 3 weeks for my refund and nobody is responding!
         My order #ORD-12345 was returned on October 1st. This is unacceptable."
    """
    complaint_category: str = Field(description="e.g. Refund, Delivery, Product Quality, Account")
    sub_category: Optional[str] = None
    urgency: UrgencyLevel
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    customer_emotion: Literal["angry", "frustrated", "disappointed", "confused", "neutral"]
    key_issue: str = Field(max_length=200, description="One-sentence summary of the core issue")
    mentioned_order_id: Optional[str] = Field(None, description="Any order/reference number mentioned")
    mentioned_amount: Optional[float] = Field(None, description="Any monetary amount mentioned (MYR)")
    days_waiting: Optional[int] = Field(None, description="How many days has customer been waiting")
    previous_contacts: Optional[int] = Field(None, description="How many times they've contacted before")
    requires_escalation: bool = Field(description="True if should be escalated to supervisor")
    suggested_action: str = Field(description="Recommended immediate action for agent")
    auto_response_safe: bool = Field(description="True if can be handled by automated response")


# ── 5. Resume / CV Parser ─────────────────────────────────────────────────────

class WorkExperience(BaseModel):
    company: str
    role: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    is_current: bool = False
    description: Optional[str] = None
    duration_years: Optional[float] = None


class Education(BaseModel):
    institution: str
    degree: Optional[str] = None
    field_of_study: Optional[str] = None
    graduation_year: Optional[int] = None
    gpa: Optional[float] = None


class ResumeSchema(BaseModel):
    """
    Extract structured candidate profile from resume/CV text.
    Useful for HR chatbots and recruitment automation.
    """
    full_name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    current_role: Optional[str] = None
    years_total_experience: Optional[float] = Field(None, ge=0, le=50)
    seniority_level: Optional[SeniorityLevel] = None
    summary: Optional[str] = Field(None, max_length=500)
    skills: list[str] = Field(default_factory=list)
    work_experience: list[WorkExperience] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    linkedin_url: Optional[str] = None
    github_url: Optional[str] = None


# ── 6. Meeting Summary Extractor ──────────────────────────────────────────────

class ActionItem(BaseModel):
    task: str
    owner: Optional[str] = None
    due_date: Optional[str] = None
    priority: Optional[Literal["low", "medium", "high"]] = None


class MeetingSummarySchema(BaseModel):
    """
    Extract structured summary from meeting transcripts or notes.

    Example input:
        "Quarterly review: Q3 targets missed by 15%. Aliya to fix pipeline
         by Friday. Team to present new model by end of month. CEO concerned
         about costs — decision on cloud budget deferred to next week."
    """
    meeting_title: Optional[str] = None
    meeting_date: Optional[str] = None
    attendees: list[str] = Field(default_factory=list)
    summary: str = Field(max_length=500, description="2-3 sentence executive summary")
    key_decisions: list[str] = Field(default_factory=list, description="Decisions made in the meeting")
    action_items: list[ActionItem] = Field(default_factory=list)
    open_issues: list[str] = Field(default_factory=list, description="Unresolved issues to follow up")
    next_meeting_date: Optional[str] = None
    sentiment: Optional[Sentiment] = None
    risk_flags: list[str] = Field(default_factory=list, description="Any risks or concerns raised")


# ── 7. Product Review Analyzer ────────────────────────────────────────────────

class ProductReviewSchema(BaseModel):
    """
    Extract structured data from product reviews.
    Works for e-commerce (Shopee, Lazada) review analysis.
    """
    overall_rating: Optional[float] = Field(None, ge=1.0, le=5.0)
    sentiment: Sentiment
    would_recommend: Optional[bool] = None
    purchase_verified: Optional[bool] = None
    product_mentioned: Optional[str] = None
    positive_aspects: list[str] = Field(default_factory=list)
    negative_aspects: list[str] = Field(default_factory=list)
    feature_mentions: list[str] = Field(default_factory=list, description="Product features mentioned")
    reviewer_type: Optional[Literal["casual", "power_user", "professional", "unknown"]] = None
    response_needed: bool = Field(description="True if seller should respond to this review")
    response_urgency: Optional[Literal["immediate", "within_day", "within_week", "not_needed"]] = None
    key_quote: Optional[str] = Field(None, max_length=200, description="Most representative quote from review")


# ── 8. Financial News Alert ───────────────────────────────────────────────────

class MarketImpact(str, Enum):
    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"


class FinancialNewsSchema(BaseModel):
    """
    Extract structured financial data from news articles.
    Used for Islamic finance monitoring and investment signals.
    """
    company_name: Optional[str] = None
    ticker_symbol: Optional[str] = None
    event_type: str = Field(
        description="e.g. Earnings, Acquisition, Leadership Change, Regulatory, Macro"
    )
    market_impact: MarketImpact
    impact_confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the impact assessment")
    financial_figures: list[str] = Field(default_factory=list)
    affected_sectors: list[str] = Field(default_factory=list)
    time_horizon: Optional[Literal["immediate", "short_term", "medium_term", "long_term"]] = None
    is_shariah_relevant: bool = Field(description="True if relevant to Islamic finance/investment")
    shariah_notes: Optional[str] = Field(None, description="Notes on Shariah compliance implications")
    key_numbers: dict = Field(default_factory=dict, description="Key numbers: {metric: value}")
    requires_analyst_review: bool = False


# ── 9. HR Query Intent ────────────────────────────────────────────────────────

class HRIntentSchema(BaseModel):
    """
    Classify and route HR chatbot queries.
    Used with RAG-based HR chatbots.

    Example input:
        "I forgot to claim my overtime last month. How do I add it now?
         Also when is the deadline for performance review?"
    """
    intent_primary: str = Field(
        description="Primary intent e.g. Leave_Application, Payroll_Query, Policy_Question, Complaint"
    )
    intent_secondary: Optional[str] = Field(None, description="Secondary intent if multi-intent")
    urgency: UrgencyLevel
    requires_personal_data: bool = Field(description="True if answering needs access to employee records")
    can_be_automated: bool = Field(description="True if can be answered from policy docs alone")
    department_to_route: Optional[str] = Field(
        None, description="Which department to route to if human needed: HR/Payroll/IT/Legal"
    )
    employee_sentiment: Literal["positive", "frustrated", "urgent", "confused", "neutral"]
    policy_topics: list[str] = Field(default_factory=list, description="Which HR policies are relevant")
    data_fields_needed: list[str] = Field(
        default_factory=list, description="Employee data fields needed to answer: leave_balance, salary, etc."
    )
    suggested_response_type: Literal["direct_answer", "policy_link", "form_required", "human_escalation"]


# ── 10. Scam/Fraud Detection ──────────────────────────────────────────────────

class ScamSchema(BaseModel):
    """
    Classify text for scam/fraud indicators.
    Extends Aliya's thesis work (Umrah/Hajj scam detection).

    Example input:
        "PAKEJ UMRAH MURAH RM1500! Tempat terhad! Transfer wang sekarang
         ke akaun 1234567890. Hubungi WhatsApp +60 11-9876 5432 SEGERA!"
    """
    is_likely_scam: bool = Field(description="True if text shows strong scam indicators")
    scam_probability: float = Field(ge=0.0, le=1.0)
    scam_type: Optional[str] = Field(
        None,
        description="e.g. Umrah_Hajj, Investment, Parcel_Scam, Love_Scam, Job_Scam, Banking"
    )
    urgency_manipulation: bool = Field(description="True if uses urgency to pressure victim")
    payment_requested: bool = Field(description="True if requests payment or bank transfer")
    payment_method: Optional[str] = None
    suspicious_contact: Optional[str] = None
    trust_signals_faked: list[str] = Field(
        default_factory=list,
        description="Fake trust signals: official logos, celebrity endorsement, etc."
    )
    red_flags: list[str] = Field(default_factory=list, description="Specific red flags detected")
    target_demographics: list[str] = Field(
        default_factory=list,
        description="Who is being targeted: muslims, elderly, job_seekers, etc."
    )
    recommended_action: Literal[
        "block_and_warn", "flag_for_review", "warn_user", "safe"
    ]
    explanation: str = Field(max_length=300, description="Brief explanation of the classification")


# ── Schema registry ───────────────────────────────────────────────────────────

ALL_SCHEMAS = {
    "job_posting": {
        "schema": JobPostingSchema,
        "description": "Extract structured data from job postings",
        "example_input": "Senior AI Engineer at Grab Singapore. SGD 10,000–15,000/month. 5 years experience. Remote friendly.",
    },
    "invoice": {
        "schema": InvoiceSchema,
        "description": "Parse Malaysian tax invoices (SST format)",
        "example_input": "TAX INVOICE INV-001. Consulting services RM 5,000. SST 8%: RM 400. Total: RM 5,400.",
    },
    "news_article": {
        "schema": NewsArticleSchema,
        "description": "Extract intelligence from news articles",
        "example_input": "Maybank Q3 profit rises 12% to RM 2.4 billion, beating analyst forecasts.",
    },
    "customer_complaint": {
        "schema": CustomerComplaintSchema,
        "description": "Classify and route customer complaints",
        "example_input": "I've been waiting 3 weeks for my refund! Order #ORD-12345. Unacceptable!",
    },
    "resume": {
        "schema": ResumeSchema,
        "description": "Parse CV/resume into structured candidate profile",
        "example_input": "Aliya Alias. AI Engineer. MSc AI (UM). 2 years experience. Python, LangChain, YOLOv8.",
    },
    "meeting_summary": {
        "schema": MeetingSummarySchema,
        "description": "Extract action items and decisions from meeting notes",
        "example_input": "Q3 review: targets missed by 15%. Aliya to fix pipeline by Friday. Budget decision deferred.",
    },
    "product_review": {
        "schema": ProductReviewSchema,
        "description": "Analyze product reviews from e-commerce platforms",
        "example_input": "4/5 stars. Great quality but shipping took 2 weeks. Product exactly as described.",
    },
    "financial_news": {
        "schema": FinancialNewsSchema,
        "description": "Extract signals from financial news (Islamic finance aware)",
        "example_input": "Maybank Islamic sukuk issuance oversubscribed 3x at RM 500 million. Strong demand.",
    },
    "hr_query": {
        "schema": HRIntentSchema,
        "description": "Classify HR chatbot queries for routing and automation",
        "example_input": "How do I apply for emergency leave? My child is sick and I need to leave immediately.",
    },
    "scam_detection": {
        "schema": ScamSchema,
        "description": "Detect scam/fraud in text (extends thesis work)",
        "example_input": "PAKEJ UMRAH RM1500 MURAH! Transfer sekarang! Tempat TERHAD! WhatsApp kami segera!",
    },
}
