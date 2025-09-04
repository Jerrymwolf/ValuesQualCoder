"""
Base Agent Framework for Values and Behavioral Coding
Provides common functionality for all LangChain agents
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Type
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
from enum import Enum

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate
from langchain.tools import BaseTool
from langchain_anthropic import ChatAnthropic
from langchain.schema import AgentAction, AgentFinish
from langchain.callbacks.base import BaseCallbackHandler

class AgentStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
    PAUSED = "paused"

@dataclass
class AgentResult:
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    execution_time: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentProgress:
    agent_name: str
    status: AgentStatus = AgentStatus.IDLE
    progress: float = 0.0
    current_task: str = ""
    results: Optional[AgentResult] = None
    start_time: Optional[datetime] = None
    last_update: datetime = field(default_factory=datetime.now)

class ValuesAgentCallback(BaseCallbackHandler):
    """Callback handler for tracking agent progress"""
    
    def __init__(self, agent_name: str, progress_tracker=None):
        self.agent_name = agent_name
        self.progress_tracker = progress_tracker
        self.step_count = 0
        self.total_steps = 0
    
    def on_agent_action(self, action: AgentAction, **kwargs) -> Any:
        """Called when agent takes an action"""
        self.step_count += 1
        if self.progress_tracker:
            progress = min(0.9, self.step_count / max(self.total_steps, 1))
            self.progress_tracker.update_progress(
                self.agent_name, 
                progress, 
                f"Executing: {action.tool}"
            )
    
    def on_agent_finish(self, finish: AgentFinish, **kwargs) -> Any:
        """Called when agent finishes"""
        if self.progress_tracker:
            self.progress_tracker.update_progress(
                self.agent_name, 
                1.0, 
                "Completed"
            )

class BaseValuesAgent(ABC):
    """Base class for all Values and Behavioral coding agents"""
    
    def __init__(
        self, 
        name: str, 
        model_name: str = "claude-3-5-sonnet-20241022",
        temperature: float = 0.1,
        max_tokens: int = 4000,
        progress_tracker=None
    ):
        self.name = name
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.progress_tracker = progress_tracker
        
        # Initialize logging
        self.logger = logging.getLogger(f"agent.{name}")
        
        # Initialize LLM
        self.llm = ChatAnthropic(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            anthropic_api_key=self._get_api_key()
        )
        
        # Initialize tools and prompt
        self.tools = self._initialize_tools()
        self.prompt = self._create_prompt()
        
        # Create agent executor
        self.agent = create_openai_tools_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            max_iterations=10,
            max_execution_time=300,  # 5 minutes timeout
            callbacks=[ValuesAgentCallback(self.name, progress_tracker)]
        )
        
        # Track agent state
        self.status = AgentStatus.IDLE
        self.current_task = ""
        self.execution_history: List[AgentResult] = []
    
    @abstractmethod
    def _initialize_tools(self) -> List[BaseTool]:
        """Initialize agent-specific tools"""
        pass
    
    @abstractmethod
    def _create_prompt(self) -> ChatPromptTemplate:
        """Create agent-specific prompt template"""
        pass
    
    @abstractmethod
    def _process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process and validate input data"""
        pass
    
    @abstractmethod
    def _post_process_result(self, result: Dict[str, Any]) -> AgentResult:
        """Post-process agent execution result"""
        pass
    
    def _get_api_key(self) -> str:
        """Get Anthropic API key from environment"""
        import os
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")
        return api_key
    
    async def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute the agent with given input"""
        start_time = datetime.now()
        
        try:
            # Update status
            self.status = AgentStatus.RUNNING
            self.current_task = input_data.get('task', 'Processing')
            
            if self.progress_tracker:
                self.progress_tracker.update_status(self.name, AgentStatus.RUNNING)
            
            # Process input
            processed_input = self._process_input(input_data)
            
            # Execute agent
            self.logger.info(f"Executing {self.name} with input: {processed_input}")
            
            raw_result = await self.agent_executor.ainvoke({
                "input": json.dumps(processed_input, indent=2)
            })
            
            # Process result
            result = self._post_process_result(raw_result)
            
            # Update status
            self.status = AgentStatus.COMPLETED
            execution_time = (datetime.now() - start_time).total_seconds()
            result.execution_time = execution_time
            
            if self.progress_tracker:
                self.progress_tracker.update_status(self.name, AgentStatus.COMPLETED, result)
            
            # Store in history
            self.execution_history.append(result)
            
            self.logger.info(f"{self.name} completed successfully in {execution_time:.2f}s")
            
            return result
            
        except Exception as e:
            # Handle error
            self.status = AgentStatus.ERROR
            execution_time = (datetime.now() - start_time).total_seconds()
            
            error_result = AgentResult(
                success=False,
                error=str(e),
                execution_time=execution_time,
                metadata={"input_data": input_data}
            )
            
            if self.progress_tracker:
                self.progress_tracker.update_status(self.name, AgentStatus.ERROR, error_result)
            
            self.execution_history.append(error_result)
            
            self.logger.error(f"{self.name} failed: {e}")
            
            return error_result
    
    def get_status(self) -> AgentProgress:
        """Get current agent status"""
        latest_result = self.execution_history[-1] if self.execution_history else None
        
        return AgentProgress(
            agent_name=self.name,
            status=self.status,
            current_task=self.current_task,
            results=latest_result
        )
    
    def reset(self):
        """Reset agent to initial state"""
        self.status = AgentStatus.IDLE
        self.current_task = ""
        # Keep execution history for analysis
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of agent execution history"""
        total_executions = len(self.execution_history)
        successful_executions = len([r for r in self.execution_history if r.success])
        
        avg_execution_time = 0
        if total_executions > 0:
            avg_execution_time = sum(r.execution_time for r in self.execution_history) / total_executions
        
        return {
            'agent_name': self.name,
            'total_executions': total_executions,
            'successful_executions': successful_executions,
            'success_rate': successful_executions / total_executions if total_executions > 0 else 0,
            'average_execution_time': round(avg_execution_time, 2),
            'last_execution': self.execution_history[-1].timestamp if self.execution_history else None
        }

class ProgressTracker:
    """Tracks progress across multiple agents"""
    
    def __init__(self):
        self.agents: Dict[str, AgentProgress] = {}
        self.callbacks: List[callable] = []
    
    def register_agent(self, agent_name: str):
        """Register an agent for tracking"""
        self.agents[agent_name] = AgentProgress(agent_name=agent_name)
    
    def update_status(self, agent_name: str, status: AgentStatus, result: Optional[AgentResult] = None):
        """Update agent status"""
        if agent_name in self.agents:
            self.agents[agent_name].status = status
            self.agents[agent_name].results = result
            self.agents[agent_name].last_update = datetime.now()
            
            if status == AgentStatus.RUNNING:
                self.agents[agent_name].start_time = datetime.now()
            
            # Notify callbacks
            for callback in self.callbacks:
                callback(agent_name, self.agents[agent_name])
    
    def update_progress(self, agent_name: str, progress: float, current_task: str):
        """Update agent progress"""
        if agent_name in self.agents:
            self.agents[agent_name].progress = progress
            self.agents[agent_name].current_task = current_task
            self.agents[agent_name].last_update = datetime.now()
            
            # Notify callbacks
            for callback in self.callbacks:
                callback(agent_name, self.agents[agent_name])
    
    def add_callback(self, callback: callable):
        """Add progress callback"""
        self.callbacks.append(callback)
    
    def get_overall_progress(self) -> Dict[str, Any]:
        """Get overall workflow progress"""
        if not self.agents:
            return {"overall_progress": 0, "status": "idle"}
        
        total_progress = sum(agent.progress for agent in self.agents.values())
        overall_progress = total_progress / len(self.agents)
        
        # Determine overall status
        statuses = [agent.status for agent in self.agents.values()]
        
        if AgentStatus.ERROR in statuses:
            overall_status = "error"
        elif AgentStatus.RUNNING in statuses:
            overall_status = "running"
        elif all(status == AgentStatus.COMPLETED for status in statuses):
            overall_status = "completed"
        else:
            overall_status = "in_progress"
        
        return {
            "overall_progress": round(overall_progress, 2),
            "status": overall_status,
            "agents": {name: {
                "status": agent.status.value,
                "progress": agent.progress,
                "current_task": agent.current_task
            } for name, agent in self.agents.items()}
        }

# Utility functions for common agent operations

def validate_text_input(text: str, min_length: int = 10, max_length: int = 10000) -> bool:
    """Validate text input for processing"""
    if not text or not isinstance(text, str):
        return False
    
    text = text.strip()
    return min_length <= len(text) <= max_length

def extract_json_from_response(response: str) -> Optional[Dict[str, Any]]:
    """Extract JSON data from agent response"""
    try:
        # Try to parse as direct JSON
        return json.loads(response)
    except json.JSONDecodeError:
        # Try to extract JSON from response text
        import re
        json_pattern = r'\{.*\}'
        match = re.search(json_pattern, response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    
    return None

def create_error_result(error_message: str, input_data: Dict[str, Any] = None) -> AgentResult:
    """Create standardized error result"""
    return AgentResult(
        success=False,
        error=error_message,
        metadata={"input_data": input_data} if input_data else {}
    )