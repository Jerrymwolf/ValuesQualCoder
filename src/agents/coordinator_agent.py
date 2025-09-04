"""
Coordinator Agent
Orchestrates the entire values and behavioral coding workflow
"""

from typing import Dict, Any, List
from langchain.prompts import ChatPromptTemplate
from langchain.tools import BaseTool, tool
from .base_agent import BaseValuesAgent, AgentResult, AgentStatus
from .mcp_tools import ValuesTaxonomyTool, DocumentProcessingTool, AnalysisTool
import json

class CoordinatorAgent(BaseValuesAgent):
    """
    Coordinator agent that manages the entire workflow:
    1. Document intake and validation
    2. Workflow planning based on document characteristics
    3. Agent orchestration and task delegation
    4. Progress monitoring and error handling
    5. Results compilation and quality assurance
    """
    
    def __init__(self, **kwargs):
        super().__init__(name="coordinator", **kwargs)
        self.workflow_state = {}
        self.active_agents = []
    
    def _initialize_tools(self) -> List[BaseTool]:
        """Initialize coordinator-specific tools"""
        return [
            self._create_workflow_planner_tool(),
            self._create_agent_orchestrator_tool(),
            self._create_quality_checker_tool(),
            DocumentProcessingTool(),
            ValuesTaxonomyTool(),
            AnalysisTool()
        ]
    
    def _create_prompt(self) -> ChatPromptTemplate:
        """Create coordinator agent prompt"""
        return ChatPromptTemplate.from_messages([
            ("system", """You are the Coordinator Agent for the Values and Behavioral Enactment Coding system.

Your primary responsibilities:
1. **Workflow Planning**: Analyze incoming documents and create optimal processing workflows
2. **Agent Orchestration**: Coordinate multiple specialized agents (segmentation, coding, validation, analysis)
3. **Resource Management**: Allocate resources and manage agent execution order
4. **Quality Assurance**: Ensure all processing steps meet quality standards
5. **Error Recovery**: Handle failures and implement recovery strategies

Available workflow stages:
- Document Segmentation (Text Segmentation Agent)
- Open Values Coding (Open Coding Agent) 
- Taxonomy Values Coding (Taxonomy Coding Agent)
- Values Validation (Validation Agent)
- Behavioral Scoring (Behavioral Coding Agent)
- Statistical Analysis (Analysis Agent)
- Report Generation (Report Generation Agent)

Guidelines:
- Always start with document analysis and segmentation
- For new taxonomies, prioritize open coding before taxonomy coding
- Ensure validation steps between major stages
- Monitor agent performance and adjust workflow as needed
- Provide clear progress updates and error handling

Return responses as structured JSON with workflow plans, agent assignments, and progress updates."""),
            ("human", "{input}")
        ])
    
    def _create_workflow_planner_tool(self) -> BaseTool:
        """Tool for planning optimal workflows"""
        @tool
        def plan_workflow(document_info: str, coding_requirements: str) -> str:
            """Plan the optimal workflow for processing a document"""
            try:
                doc_data = json.loads(document_info)
                requirements = json.loads(coding_requirements)
                
                # Analyze document characteristics
                word_count = doc_data.get('word_count', 0)
                document_type = doc_data.get('type', 'unknown')
                complexity = self._assess_document_complexity(doc_data)
                
                # Create workflow plan
                workflow_plan = {
                    'workflow_id': f"workflow_{doc_data.get('id', 'unknown')}",
                    'document_analysis': {
                        'word_count': word_count,
                        'estimated_sections': max(1, word_count // 75),
                        'complexity_level': complexity,
                        'processing_time_estimate': self._estimate_processing_time(word_count, complexity)
                    },
                    'stages': self._create_workflow_stages(requirements),
                    'resource_requirements': self._calculate_resource_requirements(word_count, complexity),
                    'quality_checkpoints': self._define_quality_checkpoints(),
                    'error_recovery_strategy': self._create_error_recovery_plan()
                }
                
                return json.dumps(workflow_plan, indent=2)
            except Exception as e:
                return json.dumps({'error': f'Workflow planning failed: {str(e)}'})
        
        return plan_workflow
    
    def _create_agent_orchestrator_tool(self) -> BaseTool:
        """Tool for orchestrating agent execution"""
        @tool
        def orchestrate_agents(workflow_plan: str, current_stage: str) -> str:
            """Orchestrate agent execution based on workflow plan and current stage"""
            try:
                plan = json.loads(workflow_plan)
                stages = plan.get('stages', [])
                
                # Find current and next stages
                current_stage_info = None
                next_stages = []
                
                for i, stage in enumerate(stages):
                    if stage['stage_name'] == current_stage:
                        current_stage_info = stage
                        # Get next executable stages
                        for j in range(i + 1, len(stages)):
                            next_stage = stages[j]
                            if self._can_execute_stage(next_stage, stages[:j]):
                                next_stages.append(next_stage)
                                break
                
                orchestration_result = {
                    'current_stage': current_stage_info,
                    'next_stages': next_stages,
                    'agent_assignments': self._assign_agents_to_stages(next_stages),
                    'execution_order': self._determine_execution_order(next_stages),
                    'parallel_execution': self._identify_parallel_stages(next_stages)
                }
                
                return json.dumps(orchestration_result, indent=2)
            except Exception as e:
                return json.dumps({'error': f'Agent orchestration failed: {str(e)}'})
        
        return orchestrate_agents
    
    def _create_quality_checker_tool(self) -> BaseTool:
        """Tool for quality assurance checks"""
        @tool
        def check_quality(stage_results: str, quality_criteria: str) -> str:
            """Perform quality checks on stage results"""
            try:
                results = json.loads(stage_results)
                criteria = json.loads(quality_criteria)
                
                quality_report = {
                    'overall_quality': 'pending',
                    'checks_performed': [],
                    'issues_found': [],
                    'recommendations': [],
                    'pass_to_next_stage': False
                }
                
                # Perform various quality checks
                if 'segmentation' in results:
                    seg_quality = self._check_segmentation_quality(results['segmentation'])
                    quality_report['checks_performed'].append('segmentation_quality')
                    if not seg_quality['passed']:
                        quality_report['issues_found'].extend(seg_quality['issues'])
                
                if 'values_coding' in results:
                    values_quality = self._check_values_coding_quality(results['values_coding'])
                    quality_report['checks_performed'].append('values_coding_quality')
                    if not values_quality['passed']:
                        quality_report['issues_found'].extend(values_quality['issues'])
                
                if 'behavioral_coding' in results:
                    behavioral_quality = self._check_behavioral_coding_quality(results['behavioral_coding'])
                    quality_report['checks_performed'].append('behavioral_coding_quality')
                    if not behavioral_quality['passed']:
                        quality_report['issues_found'].extend(behavioral_quality['issues'])
                
                # Determine overall quality
                quality_report['overall_quality'] = 'pass' if len(quality_report['issues_found']) == 0 else 'fail'
                quality_report['pass_to_next_stage'] = quality_report['overall_quality'] == 'pass'
                
                # Generate recommendations
                if quality_report['issues_found']:
                    quality_report['recommendations'] = self._generate_quality_recommendations(quality_report['issues_found'])
                
                return json.dumps(quality_report, indent=2)
            except Exception as e:
                return json.dumps({'error': f'Quality check failed: {str(e)}'})
        
        return check_quality
    
    def _process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process coordinator input"""
        required_fields = ['action', 'document_info']
        
        for field in required_fields:
            if field not in input_data:
                raise ValueError(f"Missing required field: {field}")
        
        action = input_data['action']
        valid_actions = ['plan_workflow', 'orchestrate_execution', 'monitor_progress', 'handle_error', 'finalize_results']
        
        if action not in valid_actions:
            raise ValueError(f"Invalid action: {action}. Must be one of {valid_actions}")
        
        return {
            'action': action,
            'document_info': input_data['document_info'],
            'coding_requirements': input_data.get('coding_requirements', {}),
            'current_state': input_data.get('current_state', {}),
            'agent_results': input_data.get('agent_results', {}),
            'error_info': input_data.get('error_info', {})
        }
    
    def _post_process_result(self, result: Dict[str, Any]) -> AgentResult:
        """Post-process coordinator results"""
        try:
            output = result.get('output', '')
            
            # Extract structured result
            if isinstance(output, str):
                import re
                json_match = re.search(r'\{.*\}', output, re.DOTALL)
                if json_match:
                    try:
                        parsed_result = json.loads(json_match.group())
                    except json.JSONDecodeError:
                        parsed_result = {'raw_output': output}
                else:
                    parsed_result = {'raw_output': output}
            else:
                parsed_result = output
            
            return AgentResult(
                success=True,
                data=parsed_result,
                metadata={
                    'agent_type': 'coordinator',
                    'tools_used': [step.tool for step in result.get('intermediate_steps', [])],
                    'workflow_state': self.workflow_state
                }
            )
        except Exception as e:
            return AgentResult(
                success=False,
                error=f"Result processing failed: {str(e)}",
                data={'raw_result': result}
            )
    
    # Helper methods for workflow management
    
    def _assess_document_complexity(self, doc_data: Dict[str, Any]) -> str:
        """Assess document complexity for workflow planning"""
        word_count = doc_data.get('word_count', 0)
        
        if word_count < 500:
            return 'simple'
        elif word_count < 2000:
            return 'moderate'
        else:
            return 'complex'
    
    def _estimate_processing_time(self, word_count: int, complexity: str) -> Dict[str, int]:
        """Estimate processing time for each stage"""
        base_time = {
            'simple': {'segmentation': 1, 'coding': 3, 'validation': 1, 'analysis': 1},
            'moderate': {'segmentation': 2, 'coding': 8, 'validation': 3, 'analysis': 2},
            'complex': {'segmentation': 5, 'coding': 20, 'validation': 8, 'analysis': 5}
        }
        
        return base_time.get(complexity, base_time['moderate'])
    
    def _create_workflow_stages(self, requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create workflow stages based on requirements"""
        stages = [
            {
                'stage_name': 'segmentation',
                'agent': 'text_segmentation',
                'description': 'Break document into analyzable segments',
                'dependencies': [],
                'parallel_execution': False
            }
        ]
        
        # Add coding stages based on requirements
        if requirements.get('open_coding', True):
            stages.append({
                'stage_name': 'open_coding',
                'agent': 'open_coding',
                'description': 'Identify values without taxonomic constraints',
                'dependencies': ['segmentation'],
                'parallel_execution': True
            })
        
        if requirements.get('taxonomy_coding', True):
            stages.append({
                'stage_name': 'taxonomy_coding',
                'agent': 'taxonomy_coding',
                'description': 'Map values to predefined taxonomy',
                'dependencies': ['segmentation'],
                'parallel_execution': True
            })
        
        # Add validation and analysis stages
        stages.extend([
            {
                'stage_name': 'validation',
                'agent': 'validation',
                'description': 'Compare and validate coding approaches',
                'dependencies': ['open_coding', 'taxonomy_coding'],
                'parallel_execution': False
            },
            {
                'stage_name': 'behavioral_coding',
                'agent': 'behavioral_coding',
                'description': 'Apply behavioral enactment scale',
                'dependencies': ['validation'],
                'parallel_execution': False
            },
            {
                'stage_name': 'analysis',
                'agent': 'analysis',
                'description': 'Perform statistical analysis',
                'dependencies': ['behavioral_coding'],
                'parallel_execution': False
            }
        ])
        
        return stages
    
    def _calculate_resource_requirements(self, word_count: int, complexity: str) -> Dict[str, Any]:
        """Calculate resource requirements"""
        multiplier = {'simple': 1, 'moderate': 2, 'complex': 4}.get(complexity, 2)
        
        return {
            'estimated_api_calls': (word_count // 100) * multiplier,
            'memory_requirement': 'low' if word_count < 1000 else 'medium' if word_count < 5000 else 'high',
            'parallel_agents': min(3, max(1, word_count // 1000))
        }
    
    def _define_quality_checkpoints(self) -> List[Dict[str, Any]]:
        """Define quality checkpoints"""
        return [
            {
                'checkpoint': 'segmentation_complete',
                'criteria': ['minimum_segment_size', 'maximum_segment_size', 'segment_coherence']
            },
            {
                'checkpoint': 'coding_complete',
                'criteria': ['confidence_threshold', 'value_relevance', 'consistency_check']
            },
            {
                'checkpoint': 'validation_complete',
                'criteria': ['inter_rater_agreement', 'taxonomy_coverage', 'gap_analysis']
            }
        ]
    
    def _create_error_recovery_plan(self) -> Dict[str, Any]:
        """Create error recovery strategy"""
        return {
            'retry_strategies': {
                'api_timeout': {'max_retries': 3, 'backoff_factor': 2},
                'parsing_error': {'fallback_method': 'manual_parsing'},
                'quality_failure': {'action': 'human_review_required'}
            },
            'fallback_workflows': {
                'full_automation_failure': 'semi_automated_workflow',
                'agent_unavailable': 'alternative_agent_selection'
            }
        }
    
    def _can_execute_stage(self, stage: Dict[str, Any], completed_stages: List[Dict[str, Any]]) -> bool:
        """Check if a stage can be executed"""
        dependencies = stage.get('dependencies', [])
        completed_stage_names = {s['stage_name'] for s in completed_stages}
        
        return all(dep in completed_stage_names for dep in dependencies)
    
    def _assign_agents_to_stages(self, stages: List[Dict[str, Any]]) -> Dict[str, str]:
        """Assign agents to stages"""
        return {stage['stage_name']: stage['agent'] for stage in stages}
    
    def _determine_execution_order(self, stages: List[Dict[str, Any]]) -> List[str]:
        """Determine optimal execution order"""
        # Simple topological sort based on dependencies
        ordered = []
        remaining = stages.copy()
        
        while remaining:
            ready = [s for s in remaining if all(dep in ordered for dep in s.get('dependencies', []))]
            if not ready:
                break  # Circular dependency or error
            
            next_stage = ready[0]  # Could implement priority logic here
            ordered.append(next_stage['stage_name'])
            remaining.remove(next_stage)
        
        return ordered
    
    def _identify_parallel_stages(self, stages: List[Dict[str, Any]]) -> List[List[str]]:
        """Identify stages that can run in parallel"""
        parallel_groups = []
        
        for stage in stages:
            if stage.get('parallel_execution', False):
                # Find other parallel stages with same dependencies
                parallel_group = [s['stage_name'] for s in stages 
                                 if s.get('parallel_execution', False) 
                                 and s.get('dependencies') == stage.get('dependencies')]
                
                if len(parallel_group) > 1 and parallel_group not in parallel_groups:
                    parallel_groups.append(parallel_group)
        
        return parallel_groups
    
    # Quality check methods
    
    def _check_segmentation_quality(self, segmentation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Check segmentation quality"""
        segments = segmentation_results.get('segments', [])
        issues = []
        
        if len(segments) == 0:
            issues.append('No segments created')
        
        for i, segment in enumerate(segments):
            word_count = segment.get('word_count', 0)
            if word_count < 10:
                issues.append(f'Segment {i} too short ({word_count} words)')
            if word_count > 200:
                issues.append(f'Segment {i} too long ({word_count} words)')
        
        return {'passed': len(issues) == 0, 'issues': issues}
    
    def _check_values_coding_quality(self, values_results: Dict[str, Any]) -> Dict[str, Any]:
        """Check values coding quality"""
        values = values_results.get('values', [])
        issues = []
        
        if len(values) == 0:
            issues.append('No values identified')
        
        low_confidence = [v for v in values if v.get('confidence', 0) < 0.5]
        if len(low_confidence) > len(values) * 0.3:  # >30% low confidence
            issues.append(f'Too many low confidence values: {len(low_confidence)}/{len(values)}')
        
        return {'passed': len(issues) == 0, 'issues': issues}
    
    def _check_behavioral_coding_quality(self, behavioral_results: Dict[str, Any]) -> Dict[str, Any]:
        """Check behavioral coding quality"""
        sentences = behavioral_results.get('sentences', [])
        issues = []
        
        if len(sentences) == 0:
            issues.append('No behavioral coding performed')
        
        # Check for score distribution
        scores = [s.get('behavioral_score', 0) for s in sentences]
        if all(score == 0 for score in scores):
            issues.append('All behavioral scores are neutral (0)')
        
        return {'passed': len(issues) == 0, 'issues': issues}
    
    def _generate_quality_recommendations(self, issues: List[str]) -> List[str]:
        """Generate recommendations based on quality issues"""
        recommendations = []
        
        for issue in issues:
            if 'too short' in issue.lower():
                recommendations.append('Consider merging short segments with adjacent content')
            elif 'too long' in issue.lower():
                recommendations.append('Break down long segments into smaller, coherent units')
            elif 'low confidence' in issue.lower():
                recommendations.append('Review low confidence items for potential re-coding')
            elif 'no values' in issue.lower():
                recommendations.append('Verify document contains value-relevant content')
        
        return recommendations