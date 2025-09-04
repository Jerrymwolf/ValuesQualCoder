"""
LangGraph Workflow State Management
Defines state schema and management for the agentic values coding workflow
"""

from typing import Dict, Any, List, Optional, TypedDict, Annotated
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from langchain.schema import Document
import json

# Workflow status enumeration
class WorkflowStatus(Enum):
    INITIALIZED = "initialized"
    SEGMENTING = "segmenting"
    OPEN_CODING = "open_coding"
    TAXONOMY_CODING = "taxonomy_coding"
    VALIDATING = "validating"
    BEHAVIORAL_CODING = "behavioral_coding"
    ANALYZING = "analyzing"
    REPORTING = "reporting"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"

class CodingMode(Enum):
    OPEN_ONLY = "open_only"
    TAXONOMY_ONLY = "taxonomy_only"
    DUAL_CODING = "dual_coding"
    VALIDATION_MODE = "validation_mode"

# Data structures for workflow components
@dataclass
class TextSegment:
    segment_id: str
    text: str
    start_pos: int
    end_pos: int
    word_count: int
    segment_type: str = "paragraph"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'segment_id': self.segment_id,
            'text': self.text,
            'start_pos': self.start_pos,
            'end_pos': self.end_pos,
            'word_count': self.word_count,
            'segment_type': self.segment_type,
            'metadata': self.metadata
        }

@dataclass 
class OpenValue:
    value_id: str
    value_name: str
    category: str
    confidence: float
    rationale: str
    evidence: str
    segment_id: str
    value_type: str = "explicit"  # explicit, implicit, action-based, emotional
    created_by: str = "open_coding_agent"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'value_id': self.value_id,
            'value_name': self.value_name,
            'category': self.category,
            'confidence': self.confidence,
            'rationale': self.rationale,
            'evidence': self.evidence,
            'segment_id': self.segment_id,
            'value_type': self.value_type,
            'created_by': self.created_by,
            'timestamp': self.timestamp
        }

@dataclass
class TaxonomyValue:
    value_id: str
    taxonomy_value_id: int
    value_name: str
    category: str
    confidence: float
    rationale: str
    segment_id: str
    is_custom: bool = False
    created_by: str = "taxonomy_coding_agent"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'value_id': self.value_id,
            'taxonomy_value_id': self.taxonomy_value_id,
            'value_name': self.value_name,
            'category': self.category,
            'confidence': self.confidence,
            'rationale': self.rationale,
            'segment_id': self.segment_id,
            'is_custom': self.is_custom,
            'created_by': self.created_by,
            'timestamp': self.timestamp
        }

@dataclass
class ValidationResult:
    validation_id: str
    open_value_id: Optional[str]
    taxonomy_value_id: Optional[str]
    match_type: str  # exact, fuzzy, gap, conflict
    similarity_score: float
    validation_notes: str
    needs_human_review: bool = False
    created_by: str = "validation_agent"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'validation_id': self.validation_id,
            'open_value_id': self.open_value_id,
            'taxonomy_value_id': self.taxonomy_value_id,
            'match_type': self.match_type,
            'similarity_score': self.similarity_score,
            'validation_notes': self.validation_notes,
            'needs_human_review': self.needs_human_review,
            'created_by': self.created_by,
            'timestamp': self.timestamp
        }

@dataclass
class DocumentSentence:
    sentence_id: str
    segment_id: str
    sentence_number: int
    text: str
    start_pos: int
    end_pos: int
    word_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'sentence_id': self.sentence_id,
            'segment_id': self.segment_id,
            'sentence_number': self.sentence_number,
            'text': self.text,
            'start_pos': self.start_pos,
            'end_pos': self.end_pos,
            'word_count': self.word_count,
            'metadata': self.metadata
        }

@dataclass
class BehavioralScore:
    score_id: str
    sentence_id: str
    value_id: str  # Reference to either open or taxonomy value
    behavioral_score: int  # -3 to +3
    confidence: float
    rationale: str
    scale_definition: str
    created_by: str = "behavioral_coding_agent"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'score_id': self.score_id,
            'sentence_id': self.sentence_id,
            'value_id': self.value_id,
            'behavioral_score': self.behavioral_score,
            'confidence': self.confidence,
            'rationale': self.rationale,
            'scale_definition': self.scale_definition,
            'created_by': self.created_by,
            'timestamp': self.timestamp
        }

@dataclass
class AnalysisResults:
    analysis_id: str
    frequency_analysis: Dict[str, Any]
    gap_analysis: Dict[str, Any]
    statistical_summary: Dict[str, Any]
    behavioral_analysis: Dict[str, Any]
    recommendations: List[str]
    visualizations: Dict[str, str] = field(default_factory=dict)  # Chart type -> base64 data
    created_by: str = "analysis_agent"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'analysis_id': self.analysis_id,
            'frequency_analysis': self.frequency_analysis,
            'gap_analysis': self.gap_analysis,
            'statistical_summary': self.statistical_summary,
            'behavioral_analysis': self.behavioral_analysis,
            'recommendations': self.recommendations,
            'visualizations': self.visualizations,
            'created_by': self.created_by,
            'timestamp': self.timestamp
        }

@dataclass
class WorkflowError:
    error_id: str
    stage: str
    error_type: str
    error_message: str
    error_details: Dict[str, Any]
    recovery_attempted: bool = False
    recovery_successful: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'error_id': self.error_id,
            'stage': self.stage,
            'error_type': self.error_type,
            'error_message': self.error_message,
            'error_details': self.error_details,
            'recovery_attempted': self.recovery_attempted,
            'recovery_successful': self.recovery_successful,
            'timestamp': self.timestamp
        }

# TypedDict for LangGraph state
class WorkflowState(TypedDict, total=False):
    # Document information
    document_id: str
    document_text: str
    document_metadata: Dict[str, Any]
    
    # Processing components
    segments: List[Dict[str, Any]]  # List of TextSegment.to_dict()
    open_values: List[Dict[str, Any]]  # List of OpenValue.to_dict()
    taxonomy_values: List[Dict[str, Any]]  # List of TaxonomyValue.to_dict()
    validation_results: List[Dict[str, Any]]  # List of ValidationResult.to_dict()
    sentences: List[Dict[str, Any]]  # List of DocumentSentence.to_dict()
    behavioral_scores: List[Dict[str, Any]]  # List of BehavioralScore.to_dict()
    
    # Analysis results
    analysis_results: Optional[Dict[str, Any]]  # AnalysisResults.to_dict()
    
    # Workflow control
    current_stage: str
    workflow_status: str
    progress: float
    coding_mode: str
    
    # Configuration
    user_preferences: Dict[str, Any]
    quality_thresholds: Dict[str, float]
    
    # Error handling
    errors: List[Dict[str, Any]]  # List of WorkflowError.to_dict()
    retry_count: int
    
    # Metadata
    workflow_id: str
    session_id: Optional[str]
    created_by: str
    started_at: str
    updated_at: str
    completed_at: Optional[str]

class WorkflowStateManager:
    """Manages workflow state transitions and validation"""
    
    def __init__(self):
        self.valid_transitions = self._define_valid_transitions()
        self.required_data = self._define_required_data()
    
    def _define_valid_transitions(self) -> Dict[WorkflowStatus, List[WorkflowStatus]]:
        """Define valid state transitions"""
        return {
            WorkflowStatus.INITIALIZED: [WorkflowStatus.SEGMENTING, WorkflowStatus.FAILED],
            WorkflowStatus.SEGMENTING: [WorkflowStatus.OPEN_CODING, WorkflowStatus.TAXONOMY_CODING, WorkflowStatus.FAILED],
            WorkflowStatus.OPEN_CODING: [WorkflowStatus.TAXONOMY_CODING, WorkflowStatus.VALIDATING, WorkflowStatus.BEHAVIORAL_CODING, WorkflowStatus.FAILED],
            WorkflowStatus.TAXONOMY_CODING: [WorkflowStatus.VALIDATING, WorkflowStatus.BEHAVIORAL_CODING, WorkflowStatus.FAILED],
            WorkflowStatus.VALIDATING: [WorkflowStatus.BEHAVIORAL_CODING, WorkflowStatus.ANALYZING, WorkflowStatus.FAILED],
            WorkflowStatus.BEHAVIORAL_CODING: [WorkflowStatus.ANALYZING, WorkflowStatus.FAILED],
            WorkflowStatus.ANALYZING: [WorkflowStatus.REPORTING, WorkflowStatus.FAILED],
            WorkflowStatus.REPORTING: [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED],
            WorkflowStatus.FAILED: [WorkflowStatus.INITIALIZED],  # Can restart
            WorkflowStatus.PAUSED: [WorkflowStatus.SEGMENTING, WorkflowStatus.OPEN_CODING, WorkflowStatus.TAXONOMY_CODING, 
                                   WorkflowStatus.VALIDATING, WorkflowStatus.BEHAVIORAL_CODING, WorkflowStatus.ANALYZING]
        }
    
    def _define_required_data(self) -> Dict[WorkflowStatus, List[str]]:
        """Define required data for each stage"""
        return {
            WorkflowStatus.INITIALIZED: ['document_id', 'document_text'],
            WorkflowStatus.SEGMENTING: ['document_text'],
            WorkflowStatus.OPEN_CODING: ['segments'],
            WorkflowStatus.TAXONOMY_CODING: ['segments'],
            WorkflowStatus.VALIDATING: ['open_values', 'taxonomy_values'],
            WorkflowStatus.BEHAVIORAL_CODING: ['segments', 'sentences'],
            WorkflowStatus.ANALYZING: ['open_values', 'taxonomy_values', 'behavioral_scores'],
            WorkflowStatus.REPORTING: ['analysis_results']
        }
    
    def create_initial_state(
        self,
        document_id: str,
        document_text: str,
        coding_mode: CodingMode = CodingMode.DUAL_CODING,
        user_preferences: Dict[str, Any] = None,
        session_id: str = None
    ) -> WorkflowState:
        """Create initial workflow state"""
        
        workflow_id = f"workflow_{document_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        timestamp = datetime.now().isoformat()
        
        return WorkflowState(
            # Document information
            document_id=document_id,
            document_text=document_text,
            document_metadata=self._extract_document_metadata(document_text),
            
            # Processing components (empty initially)
            segments=[],
            open_values=[],
            taxonomy_values=[],
            validation_results=[],
            sentences=[],
            behavioral_scores=[],
            
            # Analysis results
            analysis_results=None,
            
            # Workflow control
            current_stage=WorkflowStatus.INITIALIZED.value,
            workflow_status=WorkflowStatus.INITIALIZED.value,
            progress=0.0,
            coding_mode=coding_mode.value,
            
            # Configuration
            user_preferences=user_preferences or self._get_default_preferences(),
            quality_thresholds=self._get_default_quality_thresholds(),
            
            # Error handling
            errors=[],
            retry_count=0,
            
            # Metadata
            workflow_id=workflow_id,
            session_id=session_id,
            created_by="system",
            started_at=timestamp,
            updated_at=timestamp,
            completed_at=None
        )
    
    def validate_state_transition(
        self,
        current_state: WorkflowState,
        target_status: WorkflowStatus
    ) -> tuple[bool, str]:
        """Validate if state transition is allowed"""
        
        current_status = WorkflowStatus(current_state['workflow_status'])
        
        # Check if transition is valid
        if target_status not in self.valid_transitions.get(current_status, []):
            return False, f"Invalid transition from {current_status.value} to {target_status.value}"
        
        # Check if required data is present
        required_fields = self.required_data.get(target_status, [])
        for field in required_fields:
            if field not in current_state or not current_state[field]:
                return False, f"Missing required field '{field}' for {target_status.value} stage"
        
        return True, "Transition valid"
    
    def update_state(
        self,
        state: WorkflowState,
        updates: Dict[str, Any]
    ) -> WorkflowState:
        """Update workflow state with new data"""
        
        # Create a copy of the state
        new_state = state.copy()
        
        # Apply updates
        for key, value in updates.items():
            new_state[key] = value
        
        # Update timestamp
        new_state['updated_at'] = datetime.now().isoformat()
        
        # Update progress based on current stage
        new_state['progress'] = self._calculate_progress(new_state)
        
        return new_state
    
    def transition_to_stage(
        self,
        state: WorkflowState,
        target_status: WorkflowStatus,
        additional_data: Dict[str, Any] = None
    ) -> tuple[WorkflowState, bool, str]:
        """Transition workflow to a new stage"""
        
        # Validate transition
        valid, message = self.validate_state_transition(state, target_status)
        if not valid:
            return state, False, message
        
        # Prepare updates
        updates = {
            'workflow_status': target_status.value,
            'current_stage': target_status.value
        }
        
        # Add any additional data
        if additional_data:
            updates.update(additional_data)
        
        # Mark as completed if final stage
        if target_status == WorkflowStatus.COMPLETED:
            updates['completed_at'] = datetime.now().isoformat()
            updates['progress'] = 1.0
        
        # Apply updates
        new_state = self.update_state(state, updates)
        
        return new_state, True, f"Successfully transitioned to {target_status.value}"
    
    def add_error(
        self,
        state: WorkflowState,
        stage: str,
        error_type: str,
        error_message: str,
        error_details: Dict[str, Any] = None
    ) -> WorkflowState:
        """Add an error to the workflow state"""
        
        error = WorkflowError(
            error_id=f"error_{len(state['errors'])}_{datetime.now().strftime('%H%M%S')}",
            stage=stage,
            error_type=error_type,
            error_message=error_message,
            error_details=error_details or {}
        )
        
        errors = state['errors'].copy()
        errors.append(error.to_dict())
        
        updates = {
            'errors': errors,
            'retry_count': state.get('retry_count', 0) + 1
        }
        
        return self.update_state(state, updates)
    
    def add_segments(
        self,
        state: WorkflowState,
        segments: List[TextSegment]
    ) -> WorkflowState:
        """Add text segments to state"""
        
        segment_dicts = [segment.to_dict() for segment in segments]
        
        updates = {
            'segments': segment_dicts
        }
        
        return self.update_state(state, updates)
    
    def add_open_values(
        self,
        state: WorkflowState,
        open_values: List[OpenValue]
    ) -> WorkflowState:
        """Add open coding values to state"""
        
        value_dicts = [value.to_dict() for value in open_values]
        
        # Merge with existing values
        existing_values = state.get('open_values', [])
        all_values = existing_values + value_dicts
        
        updates = {
            'open_values': all_values
        }
        
        return self.update_state(state, updates)
    
    def add_taxonomy_values(
        self,
        state: WorkflowState,
        taxonomy_values: List[TaxonomyValue]
    ) -> WorkflowState:
        """Add taxonomy coding values to state"""
        
        value_dicts = [value.to_dict() for value in taxonomy_values]
        
        # Merge with existing values
        existing_values = state.get('taxonomy_values', [])
        all_values = existing_values + value_dicts
        
        updates = {
            'taxonomy_values': all_values
        }
        
        return self.update_state(state, updates)
    
    def add_validation_results(
        self,
        state: WorkflowState,
        validation_results: List[ValidationResult]
    ) -> WorkflowState:
        """Add validation results to state"""
        
        result_dicts = [result.to_dict() for result in validation_results]
        
        updates = {
            'validation_results': result_dicts
        }
        
        return self.update_state(state, updates)
    
    def add_behavioral_scores(
        self,
        state: WorkflowState,
        behavioral_scores: List[BehavioralScore]
    ) -> WorkflowState:
        """Add behavioral scores to state"""
        
        score_dicts = [score.to_dict() for score in behavioral_scores]
        
        # Merge with existing scores
        existing_scores = state.get('behavioral_scores', [])
        all_scores = existing_scores + score_dicts
        
        updates = {
            'behavioral_scores': all_scores
        }
        
        return self.update_state(state, updates)
    
    def set_analysis_results(
        self,
        state: WorkflowState,
        analysis_results: AnalysisResults
    ) -> WorkflowState:
        """Set analysis results in state"""
        
        updates = {
            'analysis_results': analysis_results.to_dict()
        }
        
        return self.update_state(state, updates)
    
    def get_state_summary(self, state: WorkflowState) -> Dict[str, Any]:
        """Get a summary of the current workflow state"""
        
        return {
            'workflow_id': state.get('workflow_id'),
            'document_id': state.get('document_id'),
            'current_stage': state.get('current_stage'),
            'workflow_status': state.get('workflow_status'),
            'progress': state.get('progress', 0.0),
            'coding_mode': state.get('coding_mode'),
            'counts': {
                'segments': len(state.get('segments', [])),
                'open_values': len(state.get('open_values', [])),
                'taxonomy_values': len(state.get('taxonomy_values', [])),
                'validation_results': len(state.get('validation_results', [])),
                'sentences': len(state.get('sentences', [])),
                'behavioral_scores': len(state.get('behavioral_scores', [])),
            },
            'has_analysis': state.get('analysis_results') is not None,
            'error_count': len(state.get('errors', [])),
            'started_at': state.get('started_at'),
            'updated_at': state.get('updated_at'),
            'completed_at': state.get('completed_at')
        }
    
    # Helper methods
    
    def _extract_document_metadata(self, document_text: str) -> Dict[str, Any]:
        """Extract metadata from document"""
        word_count = len(document_text.split())
        char_count = len(document_text)
        paragraph_count = len([p for p in document_text.split('\n\n') if p.strip()])
        
        return {
            'word_count': word_count,
            'character_count': char_count,
            'paragraph_count': paragraph_count,
            'estimated_reading_time': max(1, word_count // 200),  # Minutes
            'complexity_estimate': self._estimate_complexity(document_text)
        }
    
    def _estimate_complexity(self, text: str) -> str:
        """Estimate document complexity"""
        word_count = len(text.split())
        avg_word_length = sum(len(word) for word in text.split()) / word_count if word_count > 0 else 0
        sentence_count = len([s for s in text.split('.') if s.strip()])
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        
        if word_count < 500 and avg_sentence_length < 15:
            return 'simple'
        elif word_count < 2000 and avg_sentence_length < 25:
            return 'moderate'
        else:
            return 'complex'
    
    def _get_default_preferences(self) -> Dict[str, Any]:
        """Get default user preferences"""
        return {
            'segment_min_words': 20,
            'segment_max_words': 150,
            'confidence_threshold': 0.5,
            'auto_advance': True,
            'include_patterns': True,
            'parallel_processing': True
        }
    
    def _get_default_quality_thresholds(self) -> Dict[str, float]:
        """Get default quality thresholds"""
        return {
            'min_segment_confidence': 0.6,
            'min_value_confidence': 0.5,
            'min_behavioral_confidence': 0.4,
            'max_error_rate': 0.1,
            'min_inter_rater_agreement': 0.7
        }
    
    def _calculate_progress(self, state: WorkflowState) -> float:
        """Calculate workflow progress based on current state"""
        
        stage_weights = {
            WorkflowStatus.INITIALIZED.value: 0.0,
            WorkflowStatus.SEGMENTING.value: 0.1,
            WorkflowStatus.OPEN_CODING.value: 0.3,
            WorkflowStatus.TAXONOMY_CODING.value: 0.5,
            WorkflowStatus.VALIDATING.value: 0.7,
            WorkflowStatus.BEHAVIORAL_CODING.value: 0.85,
            WorkflowStatus.ANALYZING.value: 0.95,
            WorkflowStatus.REPORTING.value: 0.98,
            WorkflowStatus.COMPLETED.value: 1.0,
            WorkflowStatus.FAILED.value: 0.0,
            WorkflowStatus.PAUSED.value: state.get('progress', 0.0)
        }
        
        current_stage = state.get('current_stage', WorkflowStatus.INITIALIZED.value)
        base_progress = stage_weights.get(current_stage, 0.0)
        
        # Add micro-progress based on data completeness within stage
        if current_stage == WorkflowStatus.OPEN_CODING.value:
            segments_count = len(state.get('segments', []))
            coded_count = len(state.get('open_values', []))
            if segments_count > 0:
                stage_progress = min(0.2, (coded_count / segments_count) * 0.2)
                base_progress += stage_progress
        
        elif current_stage == WorkflowStatus.TAXONOMY_CODING.value:
            segments_count = len(state.get('segments', []))
            coded_count = len(state.get('taxonomy_values', []))
            if segments_count > 0:
                stage_progress = min(0.2, (coded_count / segments_count) * 0.2)
                base_progress += stage_progress
        
        return min(1.0, base_progress)

# Utility functions for state serialization
def serialize_state(state: WorkflowState) -> str:
    """Serialize workflow state to JSON string"""
    return json.dumps(state, indent=2, default=str)

def deserialize_state(state_json: str) -> WorkflowState:
    """Deserialize JSON string to workflow state"""
    return json.loads(state_json)

def validate_state_schema(state: Dict[str, Any]) -> tuple[bool, List[str]]:
    """Validate that a state dict conforms to WorkflowState schema"""
    errors = []
    
    required_fields = [
        'workflow_id', 'document_id', 'document_text', 'current_stage',
        'workflow_status', 'progress', 'coding_mode'
    ]
    
    for field in required_fields:
        if field not in state:
            errors.append(f"Missing required field: {field}")
    
    # Validate data types
    if 'progress' in state and not isinstance(state['progress'], (int, float)):
        errors.append("Progress must be a number")
    
    if 'progress' in state and not (0.0 <= state['progress'] <= 1.0):
        errors.append("Progress must be between 0.0 and 1.0")
    
    # Validate list fields
    list_fields = ['segments', 'open_values', 'taxonomy_values', 'validation_results', 
                   'sentences', 'behavioral_scores', 'errors']
    
    for field in list_fields:
        if field in state and not isinstance(state[field], list):
            errors.append(f"Field '{field}' must be a list")
    
    return len(errors) == 0, errors