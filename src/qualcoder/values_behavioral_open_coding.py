"""
Values and Behavioral Enactment Coder - Open Coding Extension
Enhanced models and services for open coding and taxonomy validation
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

from .values_behavioral_constants import CodingMode, ConfidenceLevel
from .values_behavioral_models import ValuesBehavioralException, ValidationError

logger = logging.getLogger(__name__)


class MappingType(Enum):
    """Types of mappings between open codes and taxonomy"""
    DIRECT_MATCH = "direct_match"
    BROADER_CATEGORY = "broader_category"
    TAXONOMY_GAP = "taxonomy_gap"
    TOO_SPECIFIC = "too_specific"
    NO_MATCH = "no_match"


class RecommendationType(Enum):
    """Types of taxonomy recommendations"""
    ADD_VALUE = "add_value"
    MODIFY_VALUE = "modify_value"
    MERGE_VALUES = "merge_values"
    SPLIT_VALUE = "split_value"
    RECATEGORIZE = "recategorize"


@dataclass
class OpenCodedValue:
    """Represents a value identified through open coding"""
    open_value_id: Optional[int] = None
    section_id: int = 0
    value_name: str = ""
    suggested_category: str = ""
    confidence_score: float = 0.0
    rationale: str = ""
    coded_date: Optional[datetime] = None
    coder_name: str = ""
    model_version: str = ""
    is_validated: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'open_value_id': self.open_value_id,
            'section_id': self.section_id,
            'value_name': self.value_name,
            'suggested_category': self.suggested_category,
            'confidence_score': self.confidence_score,
            'rationale': self.rationale,
            'coded_date': self.coded_date.isoformat() if self.coded_date else None,
            'coder_name': self.coder_name,
            'model_version': self.model_version,
            'is_validated': self.is_validated
        }


@dataclass
class TaxonomyMapping:
    """Represents a mapping between open-coded values and taxonomy values"""
    mapping_id: Optional[int] = None
    open_value_id: int = 0
    taxonomy_value_id: Optional[int] = None
    mapping_type: MappingType = MappingType.NO_MATCH
    confidence_score: float = 0.0
    rationale: str = ""
    created_date: Optional[datetime] = None
    created_by: str = ""
    validated_by: str = ""
    validation_date: Optional[datetime] = None
    
    # Populated from joins
    open_value_name: str = ""
    taxonomy_value_name: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'mapping_id': self.mapping_id,
            'open_value_id': self.open_value_id,
            'taxonomy_value_id': self.taxonomy_value_id,
            'mapping_type': self.mapping_type.value,
            'confidence_score': self.confidence_score,
            'rationale': self.rationale,
            'created_date': self.created_date.isoformat() if self.created_date else None,
            'created_by': self.created_by,
            'validated_by': self.validated_by,
            'validation_date': self.validation_date.isoformat() if self.validation_date else None,
            'open_value_name': self.open_value_name,
            'taxonomy_value_name': self.taxonomy_value_name
        }


@dataclass
class TaxonomyRecommendation:
    """Represents a recommendation for taxonomy improvement"""
    recommendation_id: Optional[int] = None
    recommendation_type: RecommendationType = RecommendationType.ADD_VALUE
    current_value_name: str = ""
    suggested_value_name: str = ""
    suggested_category: str = ""
    rationale: str = ""
    supporting_evidence: str = ""
    frequency_count: int = 0
    created_date: Optional[datetime] = None
    created_by: str = ""
    status: str = "pending"  # pending, approved, rejected, implemented
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'recommendation_id': self.recommendation_id,
            'recommendation_type': self.recommendation_type.value,
            'current_value_name': self.current_value_name,
            'suggested_value_name': self.suggested_value_name,
            'suggested_category': self.suggested_category,
            'rationale': self.rationale,
            'supporting_evidence': self.supporting_evidence,
            'frequency_count': self.frequency_count,
            'created_date': self.created_date.isoformat() if self.created_date else None,
            'created_by': self.created_by,
            'status': self.status
        }


@dataclass
class HybridCodingResult:
    """Represents results from hybrid coding (both open and taxonomy)"""
    section_id: int = 0
    open_values: List[OpenCodedValue] = field(default_factory=list)
    taxonomy_matches: List[TaxonomyMapping] = field(default_factory=list)
    taxonomy_gaps: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'section_id': self.section_id,
            'open_values': [v.to_dict() for v in self.open_values],
            'taxonomy_matches': [m.to_dict() for m in self.taxonomy_matches],
            'taxonomy_gaps': self.taxonomy_gaps,
            'confidence_score': self.confidence_score
        }


@dataclass
class OpenCodingStatistics:
    """Statistics for open coding analysis"""
    total_open_values: int = 0
    unique_values: int = 0
    most_frequent_values: Dict[str, int] = field(default_factory=dict)
    category_distribution: Dict[str, int] = field(default_factory=dict)
    mapping_success_rate: float = 0.0
    taxonomy_gaps_count: int = 0
    validation_coverage: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_open_values': self.total_open_values,
            'unique_values': self.unique_values,
            'most_frequent_values': self.most_frequent_values,
            'category_distribution': self.category_distribution,
            'mapping_success_rate': self.mapping_success_rate,
            'taxonomy_gaps_count': self.taxonomy_gaps_count,
            'validation_coverage': self.validation_coverage
        }


@dataclass
class TaxonomyValidationSession:
    """Represents a taxonomy validation session"""
    validation_session_id: Optional[int] = None
    session_name: str = ""
    description: str = ""
    open_coding_sessions: List[int] = field(default_factory=list)
    total_sections: int = 0
    processed_sections: int = 0
    identified_gaps: int = 0
    mapping_accuracy: float = 0.0
    created_date: Optional[datetime] = None
    created_by: str = ""
    status: str = "active"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'validation_session_id': self.validation_session_id,
            'session_name': self.session_name,
            'description': self.description,
            'open_coding_sessions': self.open_coding_sessions,
            'total_sections': self.total_sections,
            'processed_sections': self.processed_sections,
            'identified_gaps': self.identified_gaps,
            'mapping_accuracy': self.mapping_accuracy,
            'created_date': self.created_date.isoformat() if self.created_date else None,
            'created_by': self.created_by,
            'status': self.status
        }


class OpenCodingException(ValuesBehavioralException):
    """Exception for open coding operations"""
    pass


class TaxonomyValidationException(ValuesBehavioralException):
    """Exception for taxonomy validation operations"""
    pass