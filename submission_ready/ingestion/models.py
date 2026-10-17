"""Data models shared by PDF extraction and classification."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class PageRecord:
    filename: str
    page_number: int
    text: str
    extraction_method: str
    evidence: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PageClassification:
    page_number: int
    category: str
    confidence: float
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DocumentClassification:
    category: str
    confidence: float
    uncertain: bool
    evidence: list[str] = field(default_factory=list)
    page_classifications: list[PageClassification] = field(default_factory=list)
    mixed_document: bool = False
    possible_document_starts: list[int] = field(default_factory=list)
    potential_multiple_documents: bool = False
    multiple_document_evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        return result


@dataclass
class DocumentRecord:
    filename: str
    pages: list[PageRecord]
    classification: DocumentClassification
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "pages": [page.to_dict() for page in self.pages],
            "classification": self.classification.to_dict(),
            "error": self.error,
        }
