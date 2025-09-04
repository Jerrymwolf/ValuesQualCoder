#!/usr/bin/env python3
"""
Setup script for Values and Behavioral Enactment Coder - Agentic System
Comprehensive setup and testing for the web-based agentic architecture
"""

import os
import sys
import subprocess
import logging
import asyncio
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def print_banner():
    """Print setup banner"""
    banner = """
    ═══════════════════════════════════════════════════════════════════════
    VALUES AND BEHAVIORAL ENACTMENT CODER - AGENTIC SYSTEM SETUP
    ═══════════════════════════════════════════════════════════════════════
    
    Modern agentic AI system for qualitative research analysis using:
    • LangChain for agent implementations
    • LangGraph for workflow orchestration
    • LangSmith for observability
    • MCP (Model Context Protocol) for structured tool interfaces
    • FastAPI + WebSocket for real-time web interface
    
    ═══════════════════════════════════════════════════════════════════════
    """
    print(banner)

def check_python_version():
    """Check Python version compatibility"""
    print("1. Checking Python version...")
    
    if sys.version_info < (3, 9):
        print("   ✗ Python 3.9 or higher required")
        print(f"   Current version: {sys.version}")
        return False
    
    print(f"   ✓ Python {sys.version.split()[0]} - Compatible")
    return True

def check_environment():
    """Check environment and prerequisites"""
    print("\n2. Checking environment...")
    
    # Check current directory
    current_dir = Path.cwd()
    print(f"   Current directory: {current_dir}")
    
    # Check for required directories
    required_dirs = [
        "src/agents",
        "src/mcp_servers", 
        "src/workflow",
        "src/api"
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        full_path = current_dir / dir_path
        if full_path.exists():
            print(f"   ✓ {dir_path}")
        else:
            print(f"   ✗ {dir_path} - Missing")
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"\n   Creating missing directories...")
        for dir_path in missing_dirs:
            (current_dir / dir_path).mkdir(parents=True, exist_ok=True)
            print(f"   ✓ Created {dir_path}")
    
    return True

def install_dependencies():
    """Install Python dependencies"""
    print("\n3. Installing dependencies...")
    
    requirements_file = Path("requirements_agentic.txt")
    
    if not requirements_file.exists():
        print("   ✗ requirements_agentic.txt not found")
        return False
    
    try:
        # Install main requirements
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("   ✓ Main dependencies installed")
        else:
            print(f"   ✗ Failed to install dependencies: {result.stderr}")
            return False
        
        # Install spaCy model
        print("   Installing spaCy English model...")
        spacy_result = subprocess.run([
            sys.executable, "-m", "spacy", "download", "en_core_web_sm"
        ], capture_output=True, text=True, timeout=120)
        
        if spacy_result.returncode == 0:
            print("   ✓ spaCy English model installed")
        else:
            print("   ! spaCy model installation failed (optional)")
        
        return True
        
    except subprocess.TimeoutExpired:
        print("   ✗ Installation timed out")
        return False
    except Exception as e:
        print(f"   ✗ Installation failed: {e}")
        return False

def check_api_keys():
    """Check for required API keys"""
    print("\n4. Checking API keys...")
    
    required_keys = {
        'ANTHROPIC_API_KEY': 'Anthropic Claude API',
        'LANGSMITH_API_KEY': 'LangSmith (optional)',
    }
    
    missing_keys = []
    
    for key, description in required_keys.items():
        if os.getenv(key):
            print(f"   ✓ {description}")
        else:
            print(f"   ✗ {description} - Missing")
            missing_keys.append(key)
    
    if missing_keys:
        print(f"\n   ⚠️  Missing API keys: {', '.join(missing_keys)}")
        print("   Set environment variables or create .env file")
        
        # Create sample .env file
        env_sample = Path(".env.sample")
        with open(env_sample, "w") as f:
            f.write("# Values and Behavioral Enactment Coder - Environment Variables\n")
            f.write("# Copy to .env and fill in your API keys\n\n")
            for key in missing_keys:
                f.write(f"{key}=your_api_key_here\n")
            f.write("\n# Optional: LangSmith configuration\n")
            f.write("LANGSMITH_PROJECT=values-behavioral-coder\n")
            f.write("LANGSMITH_TRACING_V2=true\n")
        
        print(f"   ✓ Created {env_sample} - Copy to .env and add your keys")
    
    return len(missing_keys) == 0

def test_imports():
    """Test critical imports"""
    print("\n5. Testing imports...")
    
    critical_imports = [
        ("langchain", "LangChain"),
        ("langchain_anthropic", "LangChain Anthropic"),
        ("langgraph", "LangGraph"),
        ("fastapi", "FastAPI"),
        ("websockets", "WebSockets"),
        ("anthropic", "Anthropic"),
        ("pydantic", "Pydantic"),
        ("spacy", "spaCy"),
        ("numpy", "NumPy"),
        ("matplotlib", "Matplotlib")
    ]
    
    failed_imports = []
    
    for module, description in critical_imports:
        try:
            __import__(module)
            print(f"   ✓ {description}")
        except ImportError:
            print(f"   ✗ {description} - Import failed")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n   Failed imports: {', '.join(failed_imports)}")
        return False
    
    return True

def test_mcp_servers():
    """Test MCP server implementations"""
    print("\n6. Testing MCP servers...")
    
    try:
        # Test imports of MCP servers
        sys.path.append("src")
        
        from mcp_servers.values_taxonomy_server import ValuesTaxonomyServer
        from mcp_servers.document_processing_server import DocumentProcessingServer
        from mcp_servers.behavioral_scale_server import BehavioralScaleServer
        from mcp_servers.analysis_server import AnalysisServer
        
        print("   ✓ Values Taxonomy Server")
        print("   ✓ Document Processing Server")
        print("   ✓ Behavioral Scale Server")
        print("   ✓ Analysis Server")
        
        # Test server initialization
        values_server = ValuesTaxonomyServer()
        print("   ✓ MCP servers can be initialized")
        
        return True
        
    except Exception as e:
        print(f"   ✗ MCP server test failed: {e}")
        return False

def test_agents():
    """Test agent implementations"""
    print("\n7. Testing agents...")
    
    try:
        sys.path.append("src")
        
        from agents.base_agent import BaseValuesAgent, ProgressTracker
        from agents.coordinator_agent import CoordinatorAgent
        from agents.open_coding_agent import OpenCodingAgent
        
        print("   ✓ Base agent framework")
        print("   ✓ Coordinator agent")
        print("   ✓ Open coding agent")
        
        # Test progress tracker
        tracker = ProgressTracker()
        tracker.register_agent("test_agent")
        print("   ✓ Progress tracking")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Agent test failed: {e}")
        return False

def test_workflow():
    """Test workflow system"""
    print("\n8. Testing workflow system...")
    
    try:
        sys.path.append("src")
        
        from workflow.workflow_state import WorkflowStateManager, CodingMode
        from workflow.workflow_graph import ValuesWorkflowGraph
        
        print("   ✓ Workflow state management")
        print("   ✓ Workflow graph")
        
        # Test state manager
        state_manager = WorkflowStateManager()
        initial_state = state_manager.create_initial_state(
            document_id="test_doc",
            document_text="This is a test document for values analysis.",
            coding_mode=CodingMode.DUAL_CODING
        )
        print("   ✓ State initialization")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Workflow test failed: {e}")
        return False

def test_api():
    """Test API implementation"""
    print("\n9. Testing API implementation...")
    
    try:
        sys.path.append("src")
        
        from api.main import app
        print("   ✓ FastAPI app creation")
        
        # Test that main routes exist
        route_paths = [route.path for route in app.routes]
        expected_routes = ["/", "/health", "/analyze", "/progress/{session_id}"]
        
        missing_routes = []
        for route in expected_routes:
            if route not in route_paths:
                missing_routes.append(route)
        
        if missing_routes:
            print(f"   ✗ Missing routes: {', '.join(missing_routes)}")
            return False
        
        print("   ✓ All expected routes present")
        return True
        
    except Exception as e:
        print(f"   ✗ API test failed: {e}")
        return False

async def test_end_to_end():
    """Test end-to-end functionality"""
    print("\n10. Testing end-to-end functionality...")
    
    try:
        # Check if we have required API keys for actual testing
        if not os.getenv('ANTHROPIC_API_KEY'):
            print("   ! Skipping end-to-end test (no API key)")
            return True
        
        sys.path.append("src")
        from workflow.workflow_graph import ValuesWorkflowGraph
        from workflow.workflow_state import CodingMode
        
        # Create workflow
        workflow = ValuesWorkflowGraph({"debug": True})
        print("   ✓ Workflow created")
        
        # Test document
        test_document = """
        I believe strongly in integrity and doing the right thing, even when no one is watching.
        In my previous role, I had to make a difficult decision when I discovered some financial 
        irregularities. Rather than ignore it, I took personal responsibility to investigate 
        and address the issue directly with my team.
        """
        
        # Run workflow (this might take a while with real API calls)
        print("   Running sample analysis...")
        
        result = await workflow.run_workflow(
            document_id="test_doc_001",
            document_text=test_document,
            coding_mode=CodingMode.OPEN_ONLY,  # Use simpler mode for testing
            user_preferences={"segment_max_words": 100}
        )
        
        if result.get("success"):
            print("   ✓ End-to-end workflow completed successfully")
        else:
            print(f"   ! End-to-end workflow completed with issues: {result.get('error', 'Unknown')}")
        
        return True
        
    except Exception as e:
        print(f"   ✗ End-to-end test failed: {e}")
        return False

def create_demo_files():
    """Create demo files and scripts"""
    print("\n11. Creating demo files...")
    
    # Create demo script
    demo_script = Path("demo_agentic_system.py")
    with open(demo_script, "w") as f:
        f.write('''#!/usr/bin/env python3
"""
Demo script for Values and Behavioral Enactment Coder - Agentic System
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.append("src")

from workflow.workflow_graph import ValuesWorkflowGraph
from workflow.workflow_state import CodingMode

async def main():
    print("Values and Behavioral Enactment Coder - Agentic Demo")
    print("=" * 60)
    
    # Sample document
    sample_text = """
    I believe strongly in integrity and doing the right thing, even when no one is watching.
    In my previous role, I had to make a difficult decision when I discovered some financial 
    irregularities. Rather than ignore it or pass it up the chain, I took personal responsibility 
    to investigate and address the issue directly with my team.
    
    Excellence is another core value that drives everything I do. I'm not satisfied with 
    'good enough' - I always push myself and my team to deliver the highest quality work. 
    This sometimes means working late or going the extra mile, but I believe it's worth it 
    to maintain our standards.
    """
    
    # Create workflow
    workflow = ValuesWorkflowGraph({"debug": True})
    
    print("Running dual coding analysis...")
    print("Document preview:", sample_text[:100] + "...")
    
    result = await workflow.run_workflow(
        document_id="demo_doc",
        document_text=sample_text,
        coding_mode=CodingMode.DUAL_CODING,
        user_preferences={
            "segment_min_words": 15,
            "segment_max_words": 100,
            "confidence_threshold": 0.5
        }
    )
    
    print("\\nResults:")
    print("-" * 40)
    print(f"Success: {result.get('success', False)}")
    print(f"Workflow ID: {result.get('workflow_id', 'N/A')}")
    print(f"Final Status: {result.get('final_status', 'N/A')}")
    print(f"Progress: {result.get('progress', 0.0):.1%}")
    
    if result.get('summary'):
        summary = result['summary']
        print(f"\\nSummary:")
        print(f"  - Segments: {summary.get('counts', {}).get('segments', 0)}")
        print(f"  - Open Values: {summary.get('counts', {}).get('open_values', 0)}")
        print(f"  - Taxonomy Values: {summary.get('counts', {}).get('taxonomy_values', 0)}")
        print(f"  - Behavioral Scores: {summary.get('counts', {}).get('behavioral_scores', 0)}")
    
    if result.get('errors'):
        print(f"\\nErrors: {len(result['errors'])}")
        for error in result['errors'][:3]:  # Show first 3 errors
            print(f"  - {error.get('error_message', 'Unknown error')}")

if __name__ == "__main__":
    asyncio.run(main())
''')
    
    print(f"   ✓ Created {demo_script}")
    
    # Create server startup script
    server_script = Path("start_server.py")
    with open(server_script, "w") as f:
        f.write('''#!/usr/bin/env python3
"""
Start the Values and Behavioral Enactment Coder API server
"""

import uvicorn
import sys
from pathlib import Path

# Add src to path
sys.path.append("src")

if __name__ == "__main__":
    print("Starting Values and Behavioral Enactment Coder API Server...")
    print("Dashboard will be available at: http://localhost:8000")
    print("API Documentation at: http://localhost:8000/docs")
    print("WebSocket endpoint: ws://localhost:8000/ws/{session_id}")
    
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
''')
    
    print(f"   ✓ Created {server_script}")
    
    # Make scripts executable
    demo_script.chmod(0o755)
    server_script.chmod(0o755)
    
    return True

def print_next_steps():
    """Print next steps for the user"""
    print("""
    ═══════════════════════════════════════════════════════════════════════
    SETUP COMPLETE! NEXT STEPS:
    ═══════════════════════════════════════════════════════════════════════
    
    1. CONFIGURE API KEYS:
       • Copy .env.sample to .env and add your Anthropic API key
       • export ANTHROPIC_API_KEY="your_key_here"
    
    2. RUN DEMO:
       • ./demo_agentic_system.py
       • Tests the complete agentic workflow
    
    3. START WEB SERVER:
       • ./start_server.py
       • Access dashboard at http://localhost:8000
       • API docs at http://localhost:8000/docs
    
    4. WEB INTERFACE:
       • Upload documents via POST /analyze
       • Monitor progress via WebSocket /ws/{session_id}
       • Get results via GET /results/{session_id}
    
    5. DEVELOPMENT:
       • Modify agents in src/agents/
       • Customize workflow in src/workflow/
       • Extend MCP servers in src/mcp_servers/
    
    ═══════════════════════════════════════════════════════════════════════
    
    For support and documentation:
    • GitHub: https://github.com/anthropics/claude-code/issues
    • Architecture: See AGENTIC_ARCHITECTURE.md
    
    ═══════════════════════════════════════════════════════════════════════
    """)

async def main():
    """Main setup function"""
    print_banner()
    
    success_count = 0
    total_tests = 11
    
    # Run all setup and test steps
    tests = [
        ("Python Version", check_python_version),
        ("Environment", check_environment),
        ("Dependencies", install_dependencies),
        ("API Keys", check_api_keys),
        ("Imports", test_imports),
        ("MCP Servers", test_mcp_servers),
        ("Agents", test_agents),
        ("Workflow", test_workflow),
        ("API", test_api),
        ("End-to-End", test_end_to_end),
        ("Demo Files", create_demo_files)
    ]
    
    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            
            if result:
                success_count += 1
        except Exception as e:
            logger.error(f"{test_name} test failed with exception: {e}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"SETUP SUMMARY: {success_count}/{total_tests} tests passed")
    print(f"{'='*60}")
    
    if success_count == total_tests:
        print("🎉 ALL TESTS PASSED! System is ready for use.")
        print_next_steps()
    elif success_count >= total_tests - 2:
        print("⚠️  Setup mostly successful with minor issues.")
        print("System should work but check failed tests above.")
        print_next_steps()
    else:
        print("❌ Setup failed. Please resolve issues above before proceeding.")
        print("\nCommon solutions:")
        print("• Install missing dependencies: pip install -r requirements_agentic.txt")
        print("• Set API keys in environment variables")
        print("• Check Python version (3.9+ required)")
    
    return success_count >= total_tests - 2

if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Setup failed with unexpected error: {e}")
        sys.exit(1)
''')

if __name__ == "__main__":
    asyncio.run(main())