"""
FastAPI Backend for Values and Behavioral Enactment Coder
Provides REST API and WebSocket endpoints for the agentic workflow
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import uvicorn

from ..workflow.workflow_graph import ValuesWorkflowGraph
from ..workflow.workflow_state import CodingMode, WorkflowStatus
from ..agents.base_agent import ProgressTracker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state for workflow management
workflow_sessions: Dict[str, Dict[str, Any]] = {}
active_workflows: Dict[str, ValuesWorkflowGraph] = {}
websocket_connections: Dict[str, WebSocket] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    logger.info("Starting Values and Behavioral Enactment Coder API")
    yield
    logger.info("Shutting down Values and Behavioral Enactment Coder API")

# Create FastAPI app
app = FastAPI(
    title="Values and Behavioral Enactment Coder",
    description="Agentic AI system for qualitative values and behavioral analysis",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API

class DocumentUpload(BaseModel):
    document_id: str = Field(..., description="Unique document identifier")
    document_name: str = Field(..., description="Human-readable document name")
    document_text: str = Field(..., description="Full document text content")
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="Additional document metadata")

class AnalysisRequest(BaseModel):
    document: DocumentUpload
    coding_mode: str = Field(default="dual_coding", description="Coding mode: open_only, taxonomy_only, dual_coding, validation_mode")
    user_preferences: Optional[Dict[str, Any]] = Field(default={}, description="User preferences for analysis")
    session_id: Optional[str] = Field(default=None, description="Session identifier for grouping analyses")

class WorkflowStatus(BaseModel):
    workflow_id: str
    status: str
    progress: float
    current_stage: str
    started_at: str
    updated_at: str
    error_count: int

class AnalysisResults(BaseModel):
    workflow_id: str
    success: bool
    analysis_results: Optional[Dict[str, Any]] = None
    summary: Optional[Dict[str, Any]] = None
    errors: List[Dict[str, Any]] = []

# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        logger.info(f"WebSocket connection established for session {session_id}")
    
    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            logger.info(f"WebSocket connection closed for session {session_id}")
    
    async def send_personal_message(self, message: Dict[str, Any], session_id: str):
        if session_id in self.active_connections:
            try:
                await self.active_connections[session_id].send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Failed to send WebSocket message to {session_id}: {e}")
                self.disconnect(session_id)
    
    async def broadcast(self, message: Dict[str, Any]):
        disconnected = []
        for session_id, connection in self.active_connections.items():
            try:
                await connection.send_text(json.dumps(message))
            except:
                disconnected.append(session_id)
        
        for session_id in disconnected:
            self.disconnect(session_id)

manager = ConnectionManager()

# Progress callback for real-time updates
def create_progress_callback(session_id: str):
    """Create a progress callback function for a specific session"""
    async def progress_callback(agent_name: str, agent_progress):
        """Send progress updates via WebSocket"""
        message = {
            "type": "agent_progress",
            "agent_name": agent_name,
            "status": agent_progress.status.value if hasattr(agent_progress.status, 'value') else str(agent_progress.status),
            "progress": agent_progress.progress,
            "current_task": agent_progress.current_task,
            "timestamp": datetime.now().isoformat()
        }
        await manager.send_personal_message(message, session_id)
    
    return progress_callback

# API Endpoints

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API documentation"""
    return """
    <html>
        <head>
            <title>Values and Behavioral Enactment Coder API</title>
        </head>
        <body>
            <h1>Values and Behavioral Enactment Coder API</h1>
            <p>Agentic AI system for qualitative values and behavioral analysis</p>
            <ul>
                <li><a href="/docs">API Documentation (Swagger)</a></li>
                <li><a href="/redoc">API Documentation (ReDoc)</a></li>
                <li><a href="/health">Health Check</a></li>
            </ul>
            <h2>WebSocket Endpoint</h2>
            <p>Connect to <code>/ws/{session_id}</code> for real-time progress updates</p>
        </body>
    </html>
    """

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "active_sessions": len(workflow_sessions),
        "active_workflows": len(active_workflows),
        "websocket_connections": len(manager.active_connections)
    }

@app.post("/analyze", response_model=Dict[str, str])
async def start_analysis(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """Start a new analysis workflow"""
    
    try:
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())
        
        # Validate coding mode
        try:
            coding_mode = CodingMode(request.coding_mode)
        except ValueError:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid coding mode: {request.coding_mode}. Must be one of: {[mode.value for mode in CodingMode]}"
            )
        
        # Create workflow session
        workflow_id = f"workflow_{request.document.document_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        session_data = {
            "session_id": session_id,
            "workflow_id": workflow_id,
            "document": request.document.dict(),
            "coding_mode": coding_mode,
            "user_preferences": request.user_preferences,
            "status": "initializing",
            "created_at": datetime.now().isoformat(),
            "progress": 0.0
        }
        
        workflow_sessions[session_id] = session_data
        
        # Create workflow instance with progress tracking
        workflow_graph = ValuesWorkflowGraph({
            "debug": True,
            "max_retries": 3
        })
        
        # Set up progress callback if WebSocket is connected
        if session_id in manager.active_connections:
            progress_callback = create_progress_callback(session_id)
            workflow_graph.progress_tracker.add_callback(progress_callback)
        
        active_workflows[workflow_id] = workflow_graph
        
        # Start workflow in background
        background_tasks.add_task(
            run_workflow_background,
            workflow_id,
            session_id,
            request.document.document_text,
            request.document.document_id,
            coding_mode,
            request.user_preferences
        )
        
        logger.info(f"Started analysis workflow {workflow_id} for session {session_id}")
        
        return {
            "session_id": session_id,
            "workflow_id": workflow_id,
            "status": "started",
            "message": "Analysis workflow started. Connect to WebSocket for real-time updates."
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start analysis: {str(e)}")

@app.get("/progress/{session_id}", response_model=WorkflowStatus)
async def get_progress(session_id: str):
    """Get progress for a specific session"""
    
    if session_id not in workflow_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_data = workflow_sessions[session_id]
    workflow_id = session_data["workflow_id"]
    
    # Get progress from workflow if active
    if workflow_id in active_workflows:
        workflow = active_workflows[workflow_id]
        progress_data = workflow.get_progress(workflow_id)
        
        # Update session data
        workflow_sessions[session_id].update({
            "status": progress_data.get("status", "unknown"),
            "progress": progress_data.get("overall_progress", 0.0),
            "updated_at": datetime.now().isoformat()
        })
    
    session_data = workflow_sessions[session_id]
    
    return WorkflowStatus(
        workflow_id=workflow_id,
        status=session_data.get("status", "unknown"),
        progress=session_data.get("progress", 0.0),
        current_stage=session_data.get("current_stage", "initializing"),
        started_at=session_data.get("created_at", ""),
        updated_at=session_data.get("updated_at", ""),
        error_count=len(session_data.get("errors", []))
    )

@app.get("/results/{session_id}", response_model=AnalysisResults)
async def get_results(session_id: str):
    """Get analysis results for a specific session"""
    
    if session_id not in workflow_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_data = workflow_sessions[session_id]
    workflow_id = session_data["workflow_id"]
    
    # Check if results are available
    results = session_data.get("results")
    if not results:
        # Check if workflow is still running
        if session_data.get("status") not in ["completed", "failed"]:
            raise HTTPException(status_code=202, detail="Analysis still in progress")
        else:
            raise HTTPException(status_code=404, detail="No results available")
    
    return AnalysisResults(
        workflow_id=workflow_id,
        success=results.get("success", False),
        analysis_results=results.get("analysis_results"),
        summary=results.get("summary"),
        errors=results.get("errors", [])
    )

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and cleanup resources"""
    
    if session_id not in workflow_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Get session data
    session_data = workflow_sessions[session_id]
    workflow_id = session_data["workflow_id"]
    
    # Cleanup
    if session_id in workflow_sessions:
        del workflow_sessions[session_id]
    
    if workflow_id in active_workflows:
        del active_workflows[workflow_id]
    
    # Close WebSocket connection if active
    manager.disconnect(session_id)
    
    logger.info(f"Deleted session {session_id} and workflow {workflow_id}")
    
    return {"message": f"Session {session_id} deleted successfully"}

@app.get("/sessions")
async def list_sessions():
    """List all active sessions"""
    
    sessions = []
    for session_id, session_data in workflow_sessions.items():
        sessions.append({
            "session_id": session_id,
            "workflow_id": session_data["workflow_id"],
            "document_name": session_data["document"]["document_name"],
            "status": session_data.get("status", "unknown"),
            "progress": session_data.get("progress", 0.0),
            "created_at": session_data["created_at"],
            "coding_mode": session_data["coding_mode"].value if hasattr(session_data["coding_mode"], 'value') else str(session_data["coding_mode"])
        })
    
    return {
        "sessions": sessions,
        "total_count": len(sessions)
    }

@app.get("/taxonomy")
async def get_taxonomy():
    """Get the values taxonomy"""
    
    # Mock taxonomy data - in production, this would come from MCP server
    taxonomy = [
        {"id": 1, "name": "Integrity", "category": "Core", "description": "Acting with honesty and moral consistency"},
        {"id": 2, "name": "Excellence", "category": "Achievement", "description": "Pursuing the highest quality and standards"},
        {"id": 3, "name": "Service", "category": "Benevolence", "description": "Helping and supporting others"},
        {"id": 4, "name": "Innovation", "category": "Self-Direction", "description": "Creating new ideas and solutions"},
        {"id": 5, "name": "Teamwork", "category": "Benevolence", "description": "Working collaboratively with others"}
    ]
    
    return {
        "taxonomy": taxonomy,
        "total_values": len(taxonomy),
        "categories": list(set(v["category"] for v in taxonomy))
    }

@app.get("/behavioral-scale")
async def get_behavioral_scale():
    """Get the behavioral enactment scale definitions"""
    
    scale = [
        {"score": -3, "name": "Extraordinary Violation", "description": "Systematically undermining values"},
        {"score": -2, "name": "Active Violation", "description": "Deliberately contradicting values"},
        {"score": -1, "name": "Capitulating", "description": "Surrendering through inaction"},
        {"score": 0, "name": "Indifference", "description": "Showing no concern for values"},
        {"score": 1, "name": "Compromising", "description": "Partial, selective enactment"},
        {"score": 2, "name": "Active Enacting", "description": "Consistent, deliberate alignment"},
        {"score": 3, "name": "Extraordinary Enacting", "description": "Going above and beyond with sacrifice"}
    ]
    
    return {
        "scale": scale,
        "range": {"min": -3, "max": 3}
    }

# WebSocket endpoint
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time progress updates"""
    
    await manager.connect(websocket, session_id)
    
    # Send initial connection confirmation
    await manager.send_personal_message({
        "type": "connection_established",
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(),
        "message": "WebSocket connection established for real-time updates"
    }, session_id)
    
    try:
        while True:
            # Keep connection alive and handle any incoming messages
            data = await websocket.receive_text()
            
            # Handle ping/pong for connection health
            if data == "ping":
                await manager.send_personal_message({
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                }, session_id)
            
            # Handle status requests
            elif data == "status":
                if session_id in workflow_sessions:
                    session_data = workflow_sessions[session_id]
                    await manager.send_personal_message({
                        "type": "status_update",
                        "workflow_id": session_data["workflow_id"],
                        "status": session_data.get("status", "unknown"),
                        "progress": session_data.get("progress", 0.0),
                        "current_stage": session_data.get("current_stage", "unknown"),
                        "timestamp": datetime.now().isoformat()
                    }, session_id)
    
    except WebSocketDisconnect:
        manager.disconnect(session_id)
        logger.info(f"WebSocket disconnected for session {session_id}")

# Background task functions

async def run_workflow_background(
    workflow_id: str,
    session_id: str,
    document_text: str,
    document_id: str,
    coding_mode: CodingMode,
    user_preferences: Dict[str, Any]
):
    """Run workflow in background task"""
    
    try:
        logger.info(f"Starting background workflow {workflow_id}")
        
        # Update session status
        if session_id in workflow_sessions:
            workflow_sessions[session_id].update({
                "status": "running",
                "current_stage": "initializing"
            })
        
        # Send WebSocket update
        await manager.send_personal_message({
            "type": "workflow_started",
            "workflow_id": workflow_id,
            "status": "running",
            "message": "Workflow execution started",
            "timestamp": datetime.now().isoformat()
        }, session_id)
        
        # Get workflow instance
        workflow = active_workflows.get(workflow_id)
        if not workflow:
            raise Exception("Workflow instance not found")
        
        # Execute workflow
        result = await workflow.run_workflow(
            document_id=document_id,
            document_text=document_text,
            coding_mode=coding_mode,
            user_preferences=user_preferences,
            session_id=session_id
        )
        
        # Update session with results
        if session_id in workflow_sessions:
            workflow_sessions[session_id].update({
                "status": "completed" if result["success"] else "failed",
                "current_stage": result.get("final_status", "unknown"),
                "progress": result.get("progress", 0.0),
                "results": result,
                "completed_at": datetime.now().isoformat()
            })
        
        # Send final WebSocket update
        await manager.send_personal_message({
            "type": "workflow_completed",
            "workflow_id": workflow_id,
            "success": result["success"],
            "final_status": result.get("final_status", "unknown"),
            "progress": result.get("progress", 0.0),
            "summary": result.get("summary"),
            "timestamp": datetime.now().isoformat()
        }, session_id)
        
        logger.info(f"Completed workflow {workflow_id} with success: {result['success']}")
    
    except Exception as e:
        logger.error(f"Background workflow {workflow_id} failed: {e}")
        
        # Update session with error
        if session_id in workflow_sessions:
            workflow_sessions[session_id].update({
                "status": "failed",
                "error": str(e),
                "completed_at": datetime.now().isoformat()
            })
        
        # Send error WebSocket update
        await manager.send_personal_message({
            "type": "workflow_error",
            "workflow_id": workflow_id,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }, session_id)

# Additional utility endpoints

@app.post("/validate-document")
async def validate_document(document: DocumentUpload):
    """Validate document before analysis"""
    
    word_count = len(document.document_text.split())
    char_count = len(document.document_text)
    
    validation_results = {
        "valid": True,
        "warnings": [],
        "errors": [],
        "metadata": {
            "word_count": word_count,
            "character_count": char_count,
            "estimated_processing_time": max(1, word_count // 100),  # minutes
            "complexity": "simple" if word_count < 500 else "moderate" if word_count < 2000 else "complex"
        }
    }
    
    # Validation checks
    if word_count < 50:
        validation_results["errors"].append("Document too short (minimum 50 words)")
        validation_results["valid"] = False
    
    if word_count > 10000:
        validation_results["warnings"].append("Large document may take significant processing time")
    
    if char_count > 100000:
        validation_results["warnings"].append("Very large document - consider breaking into smaller sections")
    
    return validation_results

@app.get("/coding-modes")
async def get_coding_modes():
    """Get available coding modes"""
    
    modes = []
    for mode in CodingMode:
        modes.append({
            "value": mode.value,
            "name": mode.name,
            "description": get_coding_mode_description(mode)
        })
    
    return {"coding_modes": modes}

def get_coding_mode_description(mode: CodingMode) -> str:
    """Get description for coding mode"""
    descriptions = {
        CodingMode.OPEN_ONLY: "Open coding only - identify values without taxonomic constraints",
        CodingMode.TAXONOMY_ONLY: "Taxonomy coding only - map values to predefined taxonomy",
        CodingMode.DUAL_CODING: "Both open and taxonomy coding with validation",
        CodingMode.VALIDATION_MODE: "Focus on validating existing taxonomy against empirical data"
    }
    return descriptions.get(mode, "Unknown coding mode")

# Main application runner
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )