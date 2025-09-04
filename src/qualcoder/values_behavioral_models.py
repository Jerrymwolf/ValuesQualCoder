"""
Values and Behavioral Enactment Coder - Data Models
Contains data model classes for values and behavioral coding functionality
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

from .values_behavioral_constants import (
    CodingStage, SessionStatus, ConfidenceLevel, SectionType, 
    ExportFormat, ExportType, BEHAVIORAL_SCALE
)


@dataclass
class CoreValue:
    """Represents a core value from the taxonomy"""
    value_id: Optional[int] = None
    value_name: str = ""
    value_category: str = ""
    description: str = ""
    definition: str = ""
    created_date: Optional[datetime] = None
    is_active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'value_id': self.value_id,
            'value_name': self.value_name,
            'value_category': self.value_category,
            'description': self.description,
            'definition': self.definition,
            'created_date': self.created_date.isoformat() if self.created_date else None,
            'is_active': self.is_active
        }


@dataclass
class ValuesCodingSession:
    """Represents a values coding session for a document"""
    session_id: Optional[int] = None
    fid: int = 0  # File ID
    coder_name: str = ""
    stage: CodingStage = CodingStage.VALUES
    session_start: Optional[datetime] = None
    session_end: Optional[datetime] = None
    status: SessionStatus = SessionStatus.IN_PROGRESS
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'session_id': self.session_id,
            'fid': self.fid,
            'coder_name': self.coder_name,
            'stage': self.stage.value,
            'session_start': self.session_start.isoformat() if self.session_start else None,
            'session_end': self.session_end.isoformat() if self.session_end else None,
            'status': self.status.value,
            'notes': self.notes
        }


@dataclass
class DocumentSection:
    """Represents a section of a document for values coding"""
    section_id: Optional[int] = None
    fid: int = 0
    session_id: int = 0
    section_number: int = 0
    section_text: str = ""
    start_pos: int = 0
    end_pos: int = 0
    section_type: SectionType = SectionType.PARAGRAPH
    created_date: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'section_id': self.section_id,
            'fid': self.fid,
            'session_id': self.session_id,
            'section_number': self.section_number,
            'section_text': self.section_text,
            'start_pos': self.start_pos,
            'end_pos': self.end_pos,
            'section_type': self.section_type.value,
            'created_date': self.created_date.isoformat() if self.created_date else None
        }


@dataclass
class ClaudeValuesSuggestion:
    """Represents a Claude AI suggestion for values coding"""
    suggestion_id: Optional[int] = None
    section_id: int = 0
    value_id: int = 0
    confidence_score: float = 0.0
    rationale: str = ""
    suggested_date: Optional[datetime] = None
    model_version: str = ""
    
    # Populated from joins
    value_name: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'suggestion_id': self.suggestion_id,
            'section_id': self.section_id,
            'value_id': self.value_id,
            'value_name': self.value_name,
            'confidence_score': self.confidence_score,
            'rationale': self.rationale,
            'suggested_date': self.suggested_date.isoformat() if self.suggested_date else None,
            'model_version': self.model_version
        }


@dataclass
class ValuesCoding:
    """Represents human-validated values coding for a section"""
    coding_id: Optional[int] = None
    section_id: int = 0
    value_id: Optional[int] = None
    custom_value_name: str = ""
    is_manual_entry: bool = False
    selected_from_suggestion: bool = False
    confidence_level: Optional[ConfidenceLevel] = None
    coder_notes: str = ""
    coded_date: Optional[datetime] = None
    coder_name: str = ""
    locked_date: Optional[datetime] = None
    is_locked: bool = False
    
    # Populated from joins
    value_name: str = ""
    section_text: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'coding_id': self.coding_id,
            'section_id': self.section_id,
            'value_id': self.value_id,
            'value_name': self.value_name,
            'custom_value_name': self.custom_value_name,
            'is_manual_entry': self.is_manual_entry,
            'selected_from_suggestion': self.selected_from_suggestion,
            'confidence_level': self.confidence_level.value if self.confidence_level else None,
            'coder_notes': self.coder_notes,
            'coded_date': self.coded_date.isoformat() if self.coded_date else None,
            'coder_name': self.coder_name,
            'locked_date': self.locked_date.isoformat() if self.locked_date else None,
            'is_locked': self.is_locked,
            'section_text': self.section_text
        }


@dataclass
class DocumentSentence:
    """Represents a sentence within a section for behavioral coding"""
    sentence_id: Optional[int] = None
    section_id: int = 0
    sentence_number: int = 0
    sentence_text: str = ""
    start_pos: int = 0
    end_pos: int = 0
    created_date: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'sentence_id': self.sentence_id,
            'section_id': self.section_id,
            'sentence_number': self.sentence_number,
            'sentence_text': self.sentence_text,
            'start_pos': self.start_pos,
            'end_pos': self.end_pos,
            'created_date': self.created_date.isoformat() if self.created_date else None
        }


@dataclass
class ClaudeBehavioralSuggestion:
    """Represents a Claude AI suggestion for behavioral coding"""
    suggestion_id: Optional[int] = None
    sentence_id: int = 0
    behavioral_score: int = 0
    confidence_score: float = 0.0
    rationale: str = ""
    suggested_date: Optional[datetime] = None
    model_version: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'suggestion_id': self.suggestion_id,
            'sentence_id': self.sentence_id,
            'behavioral_score': self.behavioral_score,
            'confidence_score': self.confidence_score,
            'rationale': self.rationale,
            'suggested_date': self.suggested_date.isoformat() if self.suggested_date else None,
            'model_version': self.model_version
        }


@dataclass
class BehavioralCoding:
    """Represents human-validated behavioral coding for a sentence"""
    coding_id: Optional[int] = None
    sentence_id: int = 0
    values_coding_id: int = 0
    behavioral_score: int = 0
    selected_from_suggestion: bool = False
    confidence_level: Optional[ConfidenceLevel] = None
    coder_rationale: str = ""
    coded_date: Optional[datetime] = None
    coder_name: str = ""
    locked_date: Optional[datetime] = None
    is_locked: bool = False
    
    # Populated from joins
    sentence_text: str = ""
    selected_value: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'coding_id': self.coding_id,
            'sentence_id': self.sentence_id,
            'values_coding_id': self.values_coding_id,
            'behavioral_score': self.behavioral_score,
            'behavioral_score_name': BEHAVIORAL_SCALE.get(self.behavioral_score, {}).get('name', ''),
            'selected_from_suggestion': self.selected_from_suggestion,
            'confidence_level': self.confidence_level.value if self.confidence_level else None,
            'coder_rationale': self.coder_rationale,
            'coded_date': self.coded_date.isoformat() if self.coded_date else None,
            'coder_name': self.coder_name,
            'locked_date': self.locked_date.isoformat() if self.locked_date else None,
            'is_locked': self.is_locked,
            'sentence_text': self.sentence_text,
            'selected_value': self.selected_value
        }


@dataclass
class BehavioralScaleDefinition:
    """Represents a definition for a behavioral scale point"""
    scale_point: int = 0
    scale_name: str = ""
    short_description: str = ""
    full_description: str = ""
    examples: str = ""
    created_date: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'scale_point': self.scale_point,
            'scale_name': self.scale_name,
            'short_description': self.short_description,
            'full_description': self.full_description,
            'examples': self.examples,
            'created_date': self.created_date.isoformat() if self.created_date else None
        }


@dataclass
class CodingProgress:
    """Represents progress tracking for a coding session"""
    progress_id: Optional[int] = None
    session_id: int = 0
    fid: int = 0
    total_sections: int = 0
    sections_values_coded: int = 0
    sections_values_locked: int = 0
    total_sentences: int = 0
    sentences_behavioral_coded: int = 0
    sentences_behavioral_locked: int = 0
    stage_1_complete: bool = False
    stage_2_complete: bool = False
    last_updated: Optional[datetime] = None
    
    @property
    def values_progress_percentage(self) -> float:
        """Calculate percentage of values coding completed"""
        if self.total_sections == 0:
            return 0.0
        return (self.sections_values_coded / self.total_sections) * 100
    
    @property
    def behavioral_progress_percentage(self) -> float:
        """Calculate percentage of behavioral coding completed"""
        if self.total_sentences == 0:
            return 0.0
        return (self.sentences_behavioral_coded / self.total_sentences) * 100
    
    @property
    def overall_progress_percentage(self) -> float:
        """Calculate overall progress percentage"""
        if not self.stage_1_complete:
            return self.values_progress_percentage * 0.5
        else:
            return 50.0 + (self.behavioral_progress_percentage * 0.5)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'progress_id': self.progress_id,
            'session_id': self.session_id,
            'fid': self.fid,
            'total_sections': self.total_sections,
            'sections_values_coded': self.sections_values_coded,
            'sections_values_locked': self.sections_values_locked,
            'total_sentences': self.total_sentences,
            'sentences_behavioral_coded': self.sentences_behavioral_coded,
            'sentences_behavioral_locked': self.sentences_behavioral_locked,
            'stage_1_complete': self.stage_1_complete,
            'stage_2_complete': self.stage_2_complete,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None,
            'values_progress_percentage': self.values_progress_percentage,
            'behavioral_progress_percentage': self.behavioral_progress_percentage,
            'overall_progress_percentage': self.overall_progress_percentage
        }


@dataclass
class ExportTemplate:
    """Represents an export template configuration"""
    template_id: Optional[int] = None
    template_name: str = ""
    template_type: ExportType = ExportType.VALUES_SUMMARY
    export_format: ExportFormat = ExportFormat.CSV
    template_config: Dict[str, Any] = field(default_factory=dict)
    created_date: Optional[datetime] = None
    created_by: str = ""
    is_default: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'template_id': self.template_id,
            'template_name': self.template_name,
            'template_type': self.template_type.value,
            'export_format': self.export_format.value,
            'template_config': self.template_config,
            'created_date': self.created_date.isoformat() if self.created_date else None,
            'created_by': self.created_by,
            'is_default': self.is_default
        }


@dataclass
class CodingStatistics:
    """Statistics for values and behavioral coding analysis"""
    total_documents: int = 0
    total_sections: int = 0
    total_sentences: int = 0
    values_distribution: Dict[str, int] = field(default_factory=dict)
    behavioral_distribution: Dict[int, int] = field(default_factory=dict)
    average_behavioral_score: float = 0.0
    most_common_value: str = ""
    most_common_behavioral_score: int = 0
    inter_rater_reliability: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_documents': self.total_documents,
            'total_sections': self.total_sections,
            'total_sentences': self.total_sentences,
            'values_distribution': self.values_distribution,
            'behavioral_distribution': self.behavioral_distribution,
            'average_behavioral_score': self.average_behavioral_score,
            'most_common_value': self.most_common_value,
            'most_common_behavioral_score': self.most_common_behavioral_score,
            'inter_rater_reliability': self.inter_rater_reliability
        }


class ValuesBehavioralException(Exception):
    """Custom exception for values behavioral coding errors"""
    pass


class ValidationError(ValuesBehavioralException):
    """Exception for validation errors"""
    pass


class DatabaseError(ValuesBehavioralException):
    """Exception for database operation errors"""
    pass


class ClaudeAPIError(ValuesBehavioralException):
    """Exception for Claude API errors"""
    pass