"""Core data models for sentrysloth."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from enum import StrEnum
from functools import cached_property
from typing import Generic, TypeVar

from pydantic import BaseModel, Field, computed_field


class Severity(StrEnum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


SEVERITY_ORDER: dict[Severity, int] = {
    Severity.INFO: 0,
    Severity.LOW: 1,
    Severity.MEDIUM: 2,
    Severity.HIGH: 3,
    Severity.CRITICAL: 4,
}


class Confidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


CONFIDENCE_ORDER: dict[Confidence, int] = {
    Confidence.LOW: 0,
    Confidence.MEDIUM: 1,
    Confidence.HIGH: 2,
}


class FindingType(StrEnum):
    VULNERABILITY = "vulnerability"
    SUSPICIOUS_CHANGE = "suspicious_change"
    SECURITY_REGRESSION = "security_regression"
    HARDCODED_SECRET = "hardcoded_secret"
    DEPENDENCY_CHANGE = "dependency_change"
    CRYPTO_WEAKNESS = "crypto_weakness"
    AUTH_BYPASS = "auth_bypass"
    INPUT_VALIDATION = "input_validation"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    INFORMATION_DISCLOSURE = "information_disclosure"


class CWEEntry(BaseModel):
    id: str = Field(description="CWE ID, e.g. CWE-79")
    name: str = Field(description="CWE name, e.g. Cross-site Scripting")


class AffectedCode(BaseModel):
    file_path: str
    start_line: int
    end_line: int
    snippet: str = Field(description="Relevant code snippet from the diff")
    is_added: bool = Field(description="True if this is newly added code")


class Evidence(BaseModel):
    description: str = Field(description="What this evidence shows")
    code: AffectedCode
    reasoning: str = Field(description="Why this is security-relevant")


class Finding(BaseModel):
    """A security-relevant finding from diff analysis."""

    repo: str
    file_path: str
    hunk_signature: str = Field(description="Stable identifier for the changed hunk")
    finding_type: FindingType
    title: str
    description: str
    severity: Severity
    confidence: Confidence
    evidence: list[Evidence] = Field(min_length=1)
    cwe: list[CWEEntry] = Field(default_factory=list)
    recommendation: str = ""
    prompt_version: str = ""

    @computed_field
    @property
    def finding_id(self) -> str:
        """Stable finding identifier based on repo, file, hunk, and type.

        Uses first 16 hex chars of SHA-256 (64 bits). Collision probability is
        negligible for the expected number of findings per scan (~1e-10 at 10k findings).
        """
        raw = f"{self.repo}:{self.file_path}:{self.hunk_signature}:{self.finding_type.value}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


class DiffHunk(BaseModel):
    """A single hunk from a unified diff."""

    source_start: int
    source_length: int
    target_start: int
    target_length: int
    content: str
    header: str = ""


class DiffChunk(BaseModel):
    """A chunk of diff content ready for LLM analysis, may combine multiple hunks."""

    file_path: str
    hunks: list[DiffHunk]
    raw_diff: str
    token_estimate: int = 0
    security_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Pre-LLM security relevance score 0-1"
    )
    language: str = ""
    context: str = Field(default="", description="Surrounding code context")
    function_signatures: list[str] = Field(default_factory=list)
    truncated: bool = False


class TriageResult(BaseModel):
    """Result from the fast triage pass."""

    chunk_file_path: str
    is_security_relevant: bool
    reason: str = ""
    categories: list[str] = Field(default_factory=list)
    confidence: Confidence = Confidence.MEDIUM
    suggested_severity: Severity = Severity.INFO


class ReleaseInfo(BaseModel):
    repo_url: str
    from_ref: str
    to_ref: str
    total_files_changed: int = 0
    total_additions: int = 0
    total_deletions: int = 0


class RepoPathRole(BaseModel):
    path: str
    role: str


class RepoModuleSummary(BaseModel):
    path_prefix: str
    purpose: str


class RepoHotspot(BaseModel):
    path: str
    reason: str


class RepoEvidence(BaseModel):
    from_ref: str
    to_ref: str
    file_path: str


class RepoProfile(BaseModel):
    """Accumulated project-level context reused across release scans."""

    schema_version: str = "1"
    repo: str
    last_ref: str = ""
    overview: list[str] = Field(default_factory=list)
    tech_stack: list[str] = Field(default_factory=list)
    entrypoints: list[RepoPathRole] = Field(default_factory=list)
    modules: list[RepoModuleSummary] = Field(default_factory=list)
    security_model: list[str] = Field(default_factory=list)
    hotspots: list[RepoHotspot] = Field(default_factory=list)
    known_risks: list[str] = Field(default_factory=list)
    evidence: list[RepoEvidence] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ScanResult(BaseModel):
    """Complete result of a security scan."""

    scan_id: str
    release: ReleaseInfo
    findings: list[Finding] = Field(default_factory=list)
    triage_stats: TriageStats | None = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    prompt_version: str = ""
    llm_metrics: LLMMetrics | None = None


class TriageStats(BaseModel):
    total_chunks: int = 0
    security_relevant: int = 0
    filtered_out: int = 0
    skipped_files: int = 0


class LLMMetrics(BaseModel):
    triage_input_tokens: int = 0
    triage_output_tokens: int = 0
    analysis_input_tokens: int = 0
    analysis_output_tokens: int = 0
    total_cost_estimate: float = 0.0
    triage_latency_ms: float = 0.0
    analysis_latency_ms: float = 0.0

    def merge(self, other: LLMMetrics) -> LLMMetrics:
        """Sum all numeric fields from two metrics objects."""
        return LLMMetrics(
            **{field: getattr(self, field) + getattr(other, field) for field in self.model_fields}
        )


T = TypeVar("T")


class LLMResponse(BaseModel, Generic[T]):
    """Wrapper for LLM responses with metadata."""

    data: T
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    model: str = ""
    prompt_version: str = ""


class BaselineEntry(BaseModel):
    finding_id: str
    reason: str = ""
    suppressed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Baseline(BaseModel):
    version: str = "1"
    entries: list[BaselineEntry] = Field(default_factory=list)

    @cached_property
    def _suppressed_ids(self) -> frozenset[str]:
        return frozenset(e.finding_id for e in self.entries)

    def is_suppressed(self, finding_id: str) -> bool:
        return finding_id in self._suppressed_ids
