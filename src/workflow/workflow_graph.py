"""
LangGraph Workflow Implementation
Orchestrates the multi-agent values and behavioral coding workflow
"""

import asyncio
from typing import Dict, Any, List, Optional, Union, Callable
from datetime import datetime
import logging

from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain.schema import HumanMessage

from .workflow_state import (
    WorkflowState, WorkflowStateManager, WorkflowStatus, CodingMode,
    TextSegment, OpenValue, TaxonomyValue, ValidationResult, 
    BehavioralScore, AnalysisResults
)

from ..agents.coordinator_agent import CoordinatorAgent
from ..agents.open_coding_agent import OpenCodingAgent
from ..agents.base_agent import ProgressTracker, AgentStatus

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ValuesWorkflowGraph:
    """
    Multi-agent workflow for values and behavioral coding using LangGraph
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.state_manager = WorkflowStateManager()
        self.progress_tracker = ProgressTracker()
        
        # Initialize agents
        self.agents = self._initialize_agents()
        
        # Create the workflow graph
        self.workflow = self._create_workflow_graph()
        
        # Set up checkpointing
        self.checkpointer = MemorySaver()
        
        # Compile the graph
        self.app = self.workflow.compile(
            checkpointer=self.checkpointer,
            interrupt_before=[],  # No interruptions by default
            debug=self.config.get('debug', False)
        )
    
    def _initialize_agents(self) -> Dict[str, Any]:
        """Initialize all agents for the workflow"""
        
        # Register agents with progress tracker
        agent_names = [
            "coordinator", "text_segmentation", "open_coding", 
            "taxonomy_coding", "validation", "behavioral_coding", 
            "analysis", "reporting"
        ]
        
        for name in agent_names:
            self.progress_tracker.register_agent(name)
        
        # Create agent instances
        agents = {
            "coordinator": CoordinatorAgent(progress_tracker=self.progress_tracker),
            "open_coding": OpenCodingAgent(progress_tracker=self.progress_tracker),
            # Additional agents would be initialized here
        }
        
        return agents
    
    def _create_workflow_graph(self) -> StateGraph:
        """Create the LangGraph workflow"""
        
        # Define the graph
        workflow = StateGraph(WorkflowState)
        
        # Add nodes
        workflow.add_node("start", self._start_node)
        workflow.add_node("coordinator", self._coordinator_node)
        workflow.add_node("text_segmentation", self._text_segmentation_node)
        workflow.add_node("open_coding", self._open_coding_node)
        workflow.add_node("taxonomy_coding", self._taxonomy_coding_node)
        workflow.add_node("validation", self._validation_node)
        workflow.add_node("behavioral_coding", self._behavioral_coding_node)
        workflow.add_node("analysis", self._analysis_node)
        workflow.add_node("reporting", self._reporting_node)
        workflow.add_node("error_handler", self._error_handler_node)
        workflow.add_node("finalize", self._finalize_node)
        
        # Define edges and routing
        workflow.add_edge(START, "start")
        workflow.add_edge("start", "coordinator")
        
        # Add conditional edges from coordinator
        workflow.add_conditional_edges(
            "coordinator",
            self._route_from_coordinator,
            {
                "segment": "text_segmentation",
                "open_code": "open_coding",
                "taxonomy_code": "taxonomy_coding",
                "validate": "validation",
                "behavioral_code": "behavioral_coding",
                "analyze": "analysis",
                "report": "reporting",
                "error": "error_handler",
                "finish": "finalize"
            }
        )
        
        # Add edges from processing nodes back to coordinator
        processing_nodes = [
            "text_segmentation", "open_coding", "taxonomy_coding", 
            "validation", "behavioral_coding", "analysis", "reporting"
        ]
        
        for node in processing_nodes:
            workflow.add_conditional_edges(
                node,
                self._route_after_processing,
                {
                    "continue": "coordinator",
                    "error": "error_handler",
                    "finish": "finalize"
                }
            )
        
        # Error handler and finalize
        workflow.add_conditional_edges(
            "error_handler",
            self._route_after_error,
            {
                "retry": "coordinator",
                "fail": "finalize"
            }
        )
        
        workflow.add_edge("finalize", END)
        
        return workflow
    
    # Node implementations
    
    async def _start_node(self, state: WorkflowState) -> WorkflowState:
        """Initialize the workflow"""
        logger.info(f"Starting workflow for document {state['document_id']}")
        
        # Transition to initialized state
        new_state, success, message = self.state_manager.transition_to_stage(
            state, WorkflowStatus.INITIALIZED
        )
        
        if not success:
            logger.error(f"Failed to initialize workflow: {message}")
            return self.state_manager.add_error(
                state, "start", "initialization_error", message
            )
        
        return new_state
    
    async def _coordinator_node(self, state: WorkflowState) -> WorkflowState:
        """Coordinate workflow decisions"""
        logger.info(f"Coordinator analyzing workflow state: {state['current_stage']}")
        
        try:
            coordinator = self.agents["coordinator"]
            
            # Determine next action based on current state
            input_data = {
                'action': self._determine_coordinator_action(state),
                'document_info': {
                    'id': state['document_id'],
                    'word_count': state['document_metadata'].get('word_count', 0),
                    'complexity': state['document_metadata'].get('complexity_estimate', 'moderate')
                },
                'current_state': {
                    'stage': state['current_stage'],
                    'segments_count': len(state.get('segments', [])),
                    'open_values_count': len(state.get('open_values', [])),
                    'taxonomy_values_count': len(state.get('taxonomy_values', [])),
                    'errors': state.get('errors', [])
                },
                'coding_requirements': {
                    'coding_mode': state['coding_mode'],
                    'quality_thresholds': state['quality_thresholds']
                }
            }
            
            result = await coordinator.execute(input_data)
            
            if result.success:
                # Update state with coordinator's decisions
                updates = result.data
                return self.state_manager.update_state(state, updates)
            else:
                return self.state_manager.add_error(
                    state, "coordinator", "coordination_error", result.error
                )
        
        except Exception as e:
            logger.error(f"Coordinator node failed: {e}")
            return self.state_manager.add_error(
                state, "coordinator", "system_error", str(e)
            )
    
    async def _text_segmentation_node(self, state: WorkflowState) -> WorkflowState:
        """Segment document into analyzable chunks"""
        logger.info("Performing text segmentation")
        
        try:
            # Transition state
            new_state, success, message = self.state_manager.transition_to_stage(
                state, WorkflowStatus.SEGMENTING
            )
            
            if not success:
                return self.state_manager.add_error(
                    state, "segmentation", "state_transition_error", message
                )
            
            # Perform segmentation using document processing tools
            document_text = state['document_text']
            preferences = state['user_preferences']
            
            # Simple segmentation implementation (would use MCP tools in production)
            segments = self._segment_document(
                document_text,
                min_words=preferences.get('segment_min_words', 20),
                max_words=preferences.get('segment_max_words', 150)
            )
            
            # Add segments to state
            final_state = self.state_manager.add_segments(new_state, segments)
            
            logger.info(f"Created {len(segments)} segments")
            return final_state
        
        except Exception as e:
            logger.error(f"Text segmentation failed: {e}")
            return self.state_manager.add_error(
                state, "segmentation", "segmentation_error", str(e)
            )
    
    async def _open_coding_node(self, state: WorkflowState) -> WorkflowState:
        """Perform open coding analysis"""
        logger.info("Performing open coding analysis")
        
        try:
            # Transition state
            new_state, success, message = self.state_manager.transition_to_stage(
                state, WorkflowStatus.OPEN_CODING
            )
            
            if not success:
                return self.state_manager.add_error(
                    state, "open_coding", "state_transition_error", message
                )
            
            # Execute open coding agent
            open_coding_agent = self.agents["open_coding"]
            
            input_data = {
                'text_segments': state['segments'],
                'coding_mode': 'open',
                'confidence_threshold': state['quality_thresholds']['min_value_confidence']
            }
            
            result = await open_coding_agent.execute(input_data)
            
            if result.success:
                # Convert result data to OpenValue objects
                open_values = self._convert_to_open_values(result.data, state['segments'])
                
                # Add to state
                final_state = self.state_manager.add_open_values(new_state, open_values)
                
                logger.info(f"Identified {len(open_values)} open values")
                return final_state
            else:
                return self.state_manager.add_error(
                    new_state, "open_coding", "analysis_error", result.error
                )
        
        except Exception as e:
            logger.error(f"Open coding failed: {e}")
            return self.state_manager.add_error(
                state, "open_coding", "system_error", str(e)
            )
    
    async def _taxonomy_coding_node(self, state: WorkflowState) -> WorkflowState:
        """Perform taxonomy-based coding"""
        logger.info("Performing taxonomy coding analysis")
        
        try:
            # Transition state
            new_state, success, message = self.state_manager.transition_to_stage(
                state, WorkflowStatus.TAXONOMY_CODING
            )
            
            if not success:
                return self.state_manager.add_error(
                    state, "taxonomy_coding", "state_transition_error", message
                )
            
            # Mock taxonomy coding implementation
            taxonomy_values = self._perform_taxonomy_coding(state['segments'])
            
            # Add to state
            final_state = self.state_manager.add_taxonomy_values(new_state, taxonomy_values)
            
            logger.info(f"Identified {len(taxonomy_values)} taxonomy values")
            return final_state
        
        except Exception as e:
            logger.error(f"Taxonomy coding failed: {e}")
            return self.state_manager.add_error(
                state, "taxonomy_coding", "system_error", str(e)
            )
    
    async def _validation_node(self, state: WorkflowState) -> WorkflowState:
        """Validate and compare coding approaches"""
        logger.info("Performing validation analysis")
        
        try:
            # Transition state
            new_state, success, message = self.state_manager.transition_to_stage(
                state, WorkflowStatus.VALIDATING
            )
            
            if not success:
                return self.state_manager.add_error(
                    state, "validation", "state_transition_error", message
                )
            
            # Perform validation
            validation_results = self._perform_validation(
                state['open_values'], 
                state['taxonomy_values']
            )
            
            # Add to state
            final_state = self.state_manager.add_validation_results(new_state, validation_results)
            
            logger.info(f"Created {len(validation_results)} validation results")
            return final_state
        
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return self.state_manager.add_error(
                state, "validation", "system_error", str(e)
            )
    
    async def _behavioral_coding_node(self, state: WorkflowState) -> WorkflowState:
        """Perform behavioral enactment scoring"""
        logger.info("Performing behavioral coding")
        
        try:
            # Transition state
            new_state, success, message = self.state_manager.transition_to_stage(
                state, WorkflowStatus.BEHAVIORAL_CODING
            )
            
            if not success:
                return self.state_manager.add_error(
                    state, "behavioral_coding", "state_transition_error", message
                )
            
            # Create sentences from segments
            sentences = self._create_sentences_from_segments(state['segments'])
            
            # Perform behavioral scoring
            behavioral_scores = self._perform_behavioral_scoring(
                sentences, 
                state['open_values'] + state['taxonomy_values']
            )
            
            # Add to state
            updated_state = self.state_manager.update_state(new_state, {'sentences': [s.to_dict() for s in sentences]})
            final_state = self.state_manager.add_behavioral_scores(updated_state, behavioral_scores)
            
            logger.info(f"Created {len(behavioral_scores)} behavioral scores")
            return final_state
        
        except Exception as e:
            logger.error(f"Behavioral coding failed: {e}")
            return self.state_manager.add_error(
                state, "behavioral_coding", "system_error", str(e)
            )
    
    async def _analysis_node(self, state: WorkflowState) -> WorkflowState:
        """Perform comprehensive analysis"""
        logger.info("Performing comprehensive analysis")
        
        try:
            # Transition state
            new_state, success, message = self.state_manager.transition_to_stage(
                state, WorkflowStatus.ANALYZING
            )
            
            if not success:
                return self.state_manager.add_error(
                    state, "analysis", "state_transition_error", message
                )
            
            # Perform analysis
            analysis_results = self._perform_comprehensive_analysis(state)
            
            # Add to state
            final_state = self.state_manager.set_analysis_results(new_state, analysis_results)
            
            logger.info("Completed comprehensive analysis")
            return final_state
        
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return self.state_manager.add_error(
                state, "analysis", "system_error", str(e)
            )
    
    async def _reporting_node(self, state: WorkflowState) -> WorkflowState:
        """Generate comprehensive report"""
        logger.info("Generating comprehensive report")
        
        try:
            # Transition state
            new_state, success, message = self.state_manager.transition_to_stage(
                state, WorkflowStatus.REPORTING
            )
            
            if not success:
                return self.state_manager.add_error(
                    state, "reporting", "state_transition_error", message
                )
            
            # Generate report (mock implementation)
            report_data = self._generate_report(state)
            
            # Add report to analysis results
            if 'analysis_results' in new_state and new_state['analysis_results']:
                analysis_results = new_state['analysis_results'].copy()
                analysis_results['final_report'] = report_data
                
                final_state = self.state_manager.update_state(new_state, {
                    'analysis_results': analysis_results
                })
            else:
                final_state = new_state
            
            logger.info("Report generation completed")
            return final_state
        
        except Exception as e:
            logger.error(f"Reporting failed: {e}")
            return self.state_manager.add_error(
                state, "reporting", "system_error", str(e)
            )
    
    async def _error_handler_node(self, state: WorkflowState) -> WorkflowState:
        """Handle errors and implement recovery strategies"""
        logger.info("Handling workflow errors")
        
        errors = state.get('errors', [])
        latest_error = errors[-1] if errors else None
        
        if not latest_error:
            return state
        
        retry_count = state.get('retry_count', 0)
        max_retries = self.config.get('max_retries', 3)
        
        if retry_count < max_retries:
            # Implement retry logic
            logger.info(f"Attempting recovery, retry {retry_count + 1}/{max_retries}")
            
            # Mark error as recovery attempted
            updated_errors = errors.copy()
            updated_errors[-1]['recovery_attempted'] = True
            
            # Reset to appropriate stage for retry
            recovery_stage = self._determine_recovery_stage(latest_error)
            
            return self.state_manager.update_state(state, {
                'errors': updated_errors,
                'current_stage': recovery_stage,
                'workflow_status': recovery_stage
            })
        else:
            # Max retries exceeded, transition to failed
            logger.error(f"Max retries exceeded, marking workflow as failed")
            
            new_state, _, _ = self.state_manager.transition_to_stage(
                state, WorkflowStatus.FAILED
            )
            
            return new_state
    
    async def _finalize_node(self, state: WorkflowState) -> WorkflowState:
        """Finalize the workflow"""
        current_status = WorkflowStatus(state['workflow_status'])
        
        if current_status != WorkflowStatus.FAILED:
            # Mark as completed
            new_state, success, message = self.state_manager.transition_to_stage(
                state, WorkflowStatus.COMPLETED
            )
            
            if success:
                logger.info(f"Workflow {state['workflow_id']} completed successfully")
                return new_state
        
        logger.info(f"Workflow {state['workflow_id']} finalized with status: {current_status.value}")
        return state
    
    # Routing functions
    
    def _route_from_coordinator(self, state: WorkflowState) -> str:
        """Route from coordinator based on workflow state and decisions"""
        current_stage = state.get('current_stage')
        errors = state.get('errors', [])
        
        # Check for errors first
        if errors:
            return "error"
        
        # Route based on current stage and data completeness
        if current_stage == WorkflowStatus.INITIALIZED.value:
            return "segment"
        
        elif current_stage == WorkflowStatus.SEGMENTING.value:
            if not state.get('segments'):
                return "segment"
            
            coding_mode = CodingMode(state.get('coding_mode', CodingMode.DUAL_CODING.value))
            if coding_mode in [CodingMode.OPEN_ONLY, CodingMode.DUAL_CODING]:
                return "open_code"
            else:
                return "taxonomy_code"
        
        elif current_stage == WorkflowStatus.OPEN_CODING.value:
            coding_mode = CodingMode(state.get('coding_mode'))
            if coding_mode == CodingMode.DUAL_CODING and not state.get('taxonomy_values'):
                return "taxonomy_code"
            elif state.get('taxonomy_values'):
                return "validate"
            else:
                return "behavioral_code"
        
        elif current_stage == WorkflowStatus.TAXONOMY_CODING.value:
            if state.get('open_values'):
                return "validate"
            else:
                return "behavioral_code"
        
        elif current_stage == WorkflowStatus.VALIDATING.value:
            return "behavioral_code"
        
        elif current_stage == WorkflowStatus.BEHAVIORAL_CODING.value:
            return "analyze"
        
        elif current_stage == WorkflowStatus.ANALYZING.value:
            return "report"
        
        elif current_stage == WorkflowStatus.REPORTING.value:
            return "finish"
        
        else:
            return "finish"
    
    def _route_after_processing(self, state: WorkflowState) -> str:
        """Route after processing nodes"""
        errors = state.get('errors', [])
        
        if errors:
            # Check if latest error is from current execution
            latest_error = errors[-1]
            if latest_error['timestamp'] > state.get('updated_at', ''):
                return "error"
        
        current_stage = state.get('current_stage')
        if current_stage == WorkflowStatus.COMPLETED.value:
            return "finish"
        
        return "continue"
    
    def _route_after_error(self, state: WorkflowState) -> str:
        """Route after error handling"""
        current_status = WorkflowStatus(state.get('workflow_status'))
        
        if current_status == WorkflowStatus.FAILED:
            return "fail"
        else:
            return "retry"
    
    # Helper methods
    
    def _determine_coordinator_action(self, state: WorkflowState) -> str:
        """Determine what action the coordinator should take"""
        current_stage = state.get('current_stage')
        
        if current_stage == WorkflowStatus.INITIALIZED.value:
            return "plan_workflow"
        elif current_stage in [WorkflowStatus.SEGMENTING.value, WorkflowStatus.OPEN_CODING.value, 
                               WorkflowStatus.TAXONOMY_CODING.value, WorkflowStatus.VALIDATING.value,
                               WorkflowStatus.BEHAVIORAL_CODING.value, WorkflowStatus.ANALYZING.value]:
            return "orchestrate_execution"
        elif state.get('errors'):
            return "handle_error"
        else:
            return "monitor_progress"
    
    def _segment_document(self, text: str, min_words: int = 20, max_words: int = 150) -> List[TextSegment]:
        """Simple document segmentation"""
        paragraphs = text.split('\n\n')
        segments = []
        current_pos = 0
        
        for i, para in enumerate(paragraphs):
            para = para.strip()
            if para:
                word_count = len(para.split())
                if word_count >= min_words:
                    segment = TextSegment(
                        segment_id=f"segment_{i}",
                        text=para,
                        start_pos=current_pos,
                        end_pos=current_pos + len(para),
                        word_count=word_count,
                        segment_type="paragraph"
                    )
                    segments.append(segment)
            current_pos += len(para) + 2
        
        return segments
    
    def _convert_to_open_values(self, result_data: Dict[str, Any], segments: List[Dict[str, Any]]) -> List[OpenValue]:
        """Convert agent result to OpenValue objects"""
        open_values = []
        
        # Extract values from result data
        identified_values = result_data.get('identified_values', [])
        
        for i, value_data in enumerate(identified_values):
            open_value = OpenValue(
                value_id=f"open_{i}_{datetime.now().strftime('%H%M%S')}",
                value_name=value_data.get('value', 'Unknown'),
                category=value_data.get('category', 'Emerging'),
                confidence=value_data.get('confidence', 0.5),
                rationale=value_data.get('rationale', ''),
                evidence=value_data.get('evidence', ''),
                segment_id=value_data.get('segment_id', segments[0]['segment_id'] if segments else 'unknown'),
                value_type=value_data.get('type', 'explicit')
            )
            open_values.append(open_value)
        
        return open_values
    
    def _perform_taxonomy_coding(self, segments: List[Dict[str, Any]]) -> List[TaxonomyValue]:
        """Mock taxonomy coding implementation"""
        taxonomy_values = []
        
        predefined_values = [
            ("Integrity", "Core", "Acting with honesty and moral consistency"),
            ("Excellence", "Achievement", "Pursuing the highest quality and standards"),
            ("Service", "Benevolence", "Helping and supporting others")
        ]
        
        for i, segment in enumerate(segments):
            # Simple mock: assign first taxonomy value to each segment
            if i < len(predefined_values):
                value_name, category, description = predefined_values[i % len(predefined_values)]
                
                taxonomy_value = TaxonomyValue(
                    value_id=f"tax_{i}_{datetime.now().strftime('%H%M%S')}",
                    taxonomy_value_id=i + 1,
                    value_name=value_name,
                    category=category,
                    confidence=0.8,
                    rationale=f"Taxonomy mapping for {value_name}: {description}",
                    segment_id=segment['segment_id']
                )
                taxonomy_values.append(taxonomy_value)
        
        return taxonomy_values
    
    def _perform_validation(self, open_values: List[Dict[str, Any]], taxonomy_values: List[Dict[str, Any]]) -> List[ValidationResult]:
        """Mock validation implementation"""
        validation_results = []
        
        # Simple validation: compare value names
        open_value_names = {v['value_name'].lower(): v for v in open_values}
        taxonomy_value_names = {v['value_name'].lower(): v for v in taxonomy_values}
        
        validation_id = 0
        
        # Find matches
        for open_name, open_value in open_value_names.items():
            if open_name in taxonomy_value_names:
                # Exact match
                taxonomy_value = taxonomy_value_names[open_name]
                validation = ValidationResult(
                    validation_id=f"val_{validation_id}",
                    open_value_id=open_value['value_id'],
                    taxonomy_value_id=taxonomy_value['value_id'],
                    match_type="exact",
                    similarity_score=1.0,
                    validation_notes=f"Exact match found: {open_name}",
                    needs_human_review=False
                )
                validation_results.append(validation)
                validation_id += 1
        
        return validation_results
    
    def _create_sentences_from_segments(self, segments: List[Dict[str, Any]]) -> List[Any]:
        """Create sentences from segments"""
        from .workflow_state import DocumentSentence
        
        sentences = []
        
        for segment in segments:
            # Simple sentence splitting
            import re
            segment_sentences = re.split(r'[.!?]+', segment['text'])
            
            for i, sentence_text in enumerate(segment_sentences):
                sentence_text = sentence_text.strip()
                if sentence_text:
                    sentence = DocumentSentence(
                        sentence_id=f"{segment['segment_id']}_sent_{i}",
                        segment_id=segment['segment_id'],
                        sentence_number=i,
                        text=sentence_text,
                        start_pos=0,  # Would calculate properly in real implementation
                        end_pos=len(sentence_text),
                        word_count=len(sentence_text.split())
                    )
                    sentences.append(sentence)
        
        return sentences
    
    def _perform_behavioral_scoring(self, sentences: List[Any], values: List[Dict[str, Any]]) -> List[BehavioralScore]:
        """Mock behavioral scoring implementation"""
        behavioral_scores = []
        
        # Simple scoring based on sentence content
        for sentence in sentences:
            # Find most relevant value for this sentence
            relevant_value = values[0] if values else None
            
            if relevant_value:
                # Simple scoring logic
                sentence_lower = sentence.text.lower()
                if any(word in sentence_lower for word in ['excellent', 'outstanding', 'exceptional']):
                    score = 3
                elif any(word in sentence_lower for word in ['good', 'positive', 'helpful']):
                    score = 2
                elif any(word in sentence_lower for word in ['tried', 'attempted', 'worked']):
                    score = 1
                elif any(word in sentence_lower for word in ['failed', 'ignored', 'avoided']):
                    score = -1
                else:
                    score = 1  # Default positive
                
                behavioral_score = BehavioralScore(
                    score_id=f"score_{sentence.sentence_id}_{datetime.now().strftime('%H%M%S')}",
                    sentence_id=sentence.sentence_id,
                    value_id=relevant_value['value_id'],
                    behavioral_score=score,
                    confidence=0.7,
                    rationale=f"Scored {score} based on behavioral indicators in: '{sentence.text[:50]}...'",
                    scale_definition=f"Score {score} definition"
                )
                behavioral_scores.append(behavioral_score)
        
        return behavioral_scores
    
    def _perform_comprehensive_analysis(self, state: WorkflowState) -> AnalysisResults:
        """Perform comprehensive analysis"""
        
        # Frequency analysis
        open_values = state.get('open_values', [])
        taxonomy_values = state.get('taxonomy_values', [])
        behavioral_scores = state.get('behavioral_scores', [])
        
        frequency_analysis = {
            'open_values_frequency': self._calculate_value_frequency(open_values),
            'taxonomy_values_frequency': self._calculate_value_frequency(taxonomy_values),
            'total_unique_values': len(set([v['value_name'] for v in open_values + taxonomy_values]))
        }
        
        # Gap analysis
        gap_analysis = {
            'open_only_count': len([v for v in open_values if not any(tv['value_name'] == v['value_name'] for tv in taxonomy_values)]),
            'taxonomy_coverage': len([v for v in taxonomy_values if any(ov['value_name'] == v['value_name'] for ov in open_values)]) / max(1, len(taxonomy_values)),
            'validation_matches': len(state.get('validation_results', []))
        }
        
        # Statistical summary
        statistical_summary = {
            'total_segments': len(state.get('segments', [])),
            'total_sentences': len(state.get('sentences', [])),
            'average_behavioral_score': sum(bs['behavioral_score'] for bs in behavioral_scores) / max(1, len(behavioral_scores)),
            'confidence_distribution': self._calculate_confidence_distribution(open_values + taxonomy_values)
        }
        
        # Behavioral analysis
        behavioral_analysis = {
            'score_distribution': self._calculate_score_distribution(behavioral_scores),
            'positive_enactment_rate': len([bs for bs in behavioral_scores if bs['behavioral_score'] > 0]) / max(1, len(behavioral_scores)),
            'negative_enactment_rate': len([bs for bs in behavioral_scores if bs['behavioral_score'] < 0]) / max(1, len(behavioral_scores))
        }
        
        # Generate recommendations
        recommendations = self._generate_analysis_recommendations(
            frequency_analysis, gap_analysis, statistical_summary, behavioral_analysis
        )
        
        analysis_results = AnalysisResults(
            analysis_id=f"analysis_{state['workflow_id']}_{datetime.now().strftime('%H%M%S')}",
            frequency_analysis=frequency_analysis,
            gap_analysis=gap_analysis,
            statistical_summary=statistical_summary,
            behavioral_analysis=behavioral_analysis,
            recommendations=recommendations
        )
        
        return analysis_results
    
    def _calculate_value_frequency(self, values: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate value frequency"""
        frequency = {}
        for value in values:
            name = value['value_name']
            frequency[name] = frequency.get(name, 0) + 1
        return frequency
    
    def _calculate_confidence_distribution(self, values: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate confidence level distribution"""
        distribution = {'high': 0, 'medium': 0, 'low': 0}
        
        for value in values:
            confidence = value.get('confidence', 0.5)
            if confidence >= 0.8:
                distribution['high'] += 1
            elif confidence >= 0.5:
                distribution['medium'] += 1
            else:
                distribution['low'] += 1
        
        return distribution
    
    def _calculate_score_distribution(self, behavioral_scores: List[Dict[str, Any]]) -> Dict[int, int]:
        """Calculate behavioral score distribution"""
        distribution = {}
        for score_data in behavioral_scores:
            score = score_data['behavioral_score']
            distribution[score] = distribution.get(score, 0) + 1
        return distribution
    
    def _generate_analysis_recommendations(self, frequency_analysis, gap_analysis, statistical_summary, behavioral_analysis) -> List[str]:
        """Generate analysis recommendations"""
        recommendations = []
        
        if gap_analysis['taxonomy_coverage'] < 0.7:
            recommendations.append("Consider expanding taxonomy to improve coverage of identified values")
        
        if behavioral_analysis['negative_enactment_rate'] > 0.3:
            recommendations.append("High rate of negative behavioral enactment detected - review for intervention opportunities")
        
        if statistical_summary['confidence_distribution']['low'] > statistical_summary['confidence_distribution']['high']:
            recommendations.append("Many low-confidence value identifications - consider additional coding passes")
        
        return recommendations
    
    def _generate_report(self, state: WorkflowState) -> Dict[str, Any]:
        """Generate comprehensive report"""
        summary = self.state_manager.get_state_summary(state)
        
        return {
            'workflow_summary': summary,
            'executive_summary': self._create_executive_summary(state),
            'detailed_findings': self._create_detailed_findings(state),
            'recommendations': state.get('analysis_results', {}).get('recommendations', []),
            'generated_at': datetime.now().isoformat()
        }
    
    def _create_executive_summary(self, state: WorkflowState) -> str:
        """Create executive summary"""
        segments_count = len(state.get('segments', []))
        values_count = len(state.get('open_values', [])) + len(state.get('taxonomy_values', []))
        behavioral_scores = state.get('behavioral_scores', [])
        
        avg_score = sum(bs['behavioral_score'] for bs in behavioral_scores) / max(1, len(behavioral_scores))
        
        return f"""
        Analysis completed for document {state['document_id']}. 
        Processed {segments_count} text segments and identified {values_count} distinct values.
        Behavioral analysis of {len(behavioral_scores)} sentences yielded an average enactment score of {avg_score:.2f}.
        """
    
    def _create_detailed_findings(self, state: WorkflowState) -> Dict[str, Any]:
        """Create detailed findings"""
        return {
            'segmentation_results': {
                'total_segments': len(state.get('segments', [])),
                'average_segment_length': sum(s['word_count'] for s in state.get('segments', [])) / max(1, len(state.get('segments', [])))
            },
            'values_identification': {
                'open_coding_values': len(state.get('open_values', [])),
                'taxonomy_values': len(state.get('taxonomy_values', [])),
                'validation_matches': len(state.get('validation_results', []))
            },
            'behavioral_scoring': {
                'total_sentences_scored': len(state.get('behavioral_scores', [])),
                'score_range': self._get_score_range(state.get('behavioral_scores', []))
            }
        }
    
    def _get_score_range(self, behavioral_scores: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get behavioral score range"""
        if not behavioral_scores:
            return {'min': 0, 'max': 0}
        
        scores = [bs['behavioral_score'] for bs in behavioral_scores]
        return {'min': min(scores), 'max': max(scores)}
    
    def _determine_recovery_stage(self, error: Dict[str, Any]) -> str:
        """Determine what stage to retry from after error"""
        error_stage = error.get('stage', 'unknown')
        
        # Map error stages to recovery stages
        recovery_mapping = {
            'segmentation': WorkflowStatus.SEGMENTING.value,
            'open_coding': WorkflowStatus.OPEN_CODING.value,
            'taxonomy_coding': WorkflowStatus.TAXONOMY_CODING.value,
            'validation': WorkflowStatus.VALIDATING.value,
            'behavioral_coding': WorkflowStatus.BEHAVIORAL_CODING.value,
            'analysis': WorkflowStatus.ANALYZING.value,
            'reporting': WorkflowStatus.REPORTING.value
        }
        
        return recovery_mapping.get(error_stage, WorkflowStatus.INITIALIZED.value)
    
    # Public methods for workflow execution
    
    async def run_workflow(
        self,
        document_id: str,
        document_text: str,
        coding_mode: CodingMode = CodingMode.DUAL_CODING,
        user_preferences: Dict[str, Any] = None,
        session_id: str = None
    ) -> Dict[str, Any]:
        """Run the complete workflow"""
        
        # Create initial state
        initial_state = self.state_manager.create_initial_state(
            document_id=document_id,
            document_text=document_text,
            coding_mode=coding_mode,
            user_preferences=user_preferences,
            session_id=session_id
        )
        
        # Configure thread for this workflow run
        thread_config = {
            "configurable": {
                "thread_id": initial_state['workflow_id']
            }
        }
        
        try:
            # Stream the workflow execution
            final_state = None
            async for event in self.app.astream(initial_state, config=thread_config):
                logger.debug(f"Workflow event: {event}")
                
                # Extract the latest state
                for node_name, node_state in event.items():
                    if isinstance(node_state, dict) and 'workflow_id' in node_state:
                        final_state = node_state
            
            if final_state:
                # Return workflow summary
                return {
                    'success': final_state.get('workflow_status') == WorkflowStatus.COMPLETED.value,
                    'workflow_id': final_state['workflow_id'],
                    'final_status': final_state.get('workflow_status'),
                    'progress': final_state.get('progress', 0.0),
                    'summary': self.state_manager.get_state_summary(final_state),
                    'analysis_results': final_state.get('analysis_results'),
                    'errors': final_state.get('errors', [])
                }
            else:
                return {
                    'success': False,
                    'error': 'Workflow execution failed - no final state returned'
                }
        
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'workflow_id': initial_state['workflow_id']
            }
    
    def get_progress(self, workflow_id: str) -> Dict[str, Any]:
        """Get current progress for a workflow"""
        return self.progress_tracker.get_overall_progress()