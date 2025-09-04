"""
MCP Tools Integration
Wraps MCP server functionality as LangChain tools
"""

from typing import Dict, Any, List, Optional
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
import json
import asyncio
import subprocess
import os
from pathlib import Path

class ValuesTaxonomyInput(BaseModel):
    """Input schema for Values Taxonomy tool"""
    action: str = Field(description="Action to perform: get_taxonomy, search_category, add_custom, get_statistics")
    category: Optional[str] = Field(None, description="Value category for filtering")
    value_name: Optional[str] = Field(None, description="Name of custom value to add")
    description: Optional[str] = Field(None, description="Description of custom value")
    created_by: Optional[str] = Field(None, description="Creator of custom value")

class DocumentProcessingInput(BaseModel):
    """Input schema for Document Processing tool"""
    action: str = Field(description="Action: segment_document, parse_sentences, extract_contexts, validate_segments, get_metadata")
    text: str = Field(description="Text to process")
    segment_type: Optional[str] = Field("paragraph", description="Segmentation type: paragraph, sentence, smart")
    min_words: Optional[int] = Field(20, description="Minimum words per segment")
    max_words: Optional[int] = Field(150, description="Maximum words per segment")
    target_phrases: Optional[List[str]] = Field(None, description="Phrases to extract context for")

class BehavioralScaleInput(BaseModel):
    """Input schema for Behavioral Scale tool"""
    action: str = Field(description="Action: get_definitions, score_behavior, get_examples, validate_scoring, calculate_reliability")
    sentence_text: Optional[str] = Field(None, description="Sentence to score")
    value_name: Optional[str] = Field(None, description="Value being enacted")
    behavioral_score: Optional[int] = Field(None, description="Proposed behavioral score (-3 to 3)")
    scorer_a_scores: Optional[List[int]] = Field(None, description="First scorer's scores for reliability")
    scorer_b_scores: Optional[List[int]] = Field(None, description="Second scorer's scores for reliability")

class AnalysisInput(BaseModel):
    """Input schema for Analysis tool"""
    action: str = Field(description="Action: calculate_frequencies, gap_analysis, generate_statistics, create_visualizations, export_results")
    session_id: Optional[int] = Field(None, description="Session ID for filtering")
    open_coded_values: Optional[List[str]] = Field(None, description="Open coded values for gap analysis")
    taxonomy_values: Optional[List[str]] = Field(None, description="Taxonomy values for gap analysis")
    data_type: Optional[str] = Field(None, description="Visualization type")
    format_type: Optional[str] = Field("json", description="Export format")

class ValuesTaxonomyTool(BaseTool):
    """Tool for interacting with Values Taxonomy MCP Server"""
    
    name = "values_taxonomy"
    description = "Access values taxonomy, add custom values, and get usage statistics"
    args_schema = ValuesTaxonomyInput
    
    def _run(self, action: str, **kwargs) -> str:
        """Run the tool synchronously"""
        return asyncio.run(self._arun(action, **kwargs))
    
    async def _arun(self, action: str, **kwargs) -> str:
        """Run the tool asynchronously"""
        try:
            server_path = Path(__file__).parent.parent / "mcp_servers" / "values_taxonomy_server.py"
            
            if action == "get_taxonomy":
                result = await self._call_mcp_server(server_path, "get_taxonomy_values", {
                    "category": kwargs.get("category")
                })
            elif action == "search_category":
                result = await self._call_mcp_server(server_path, "search_values_by_category", {
                    "category": kwargs.get("category", "")
                })
            elif action == "add_custom":
                result = await self._call_mcp_server(server_path, "add_custom_value", {
                    "value_name": kwargs.get("value_name", ""),
                    "category": kwargs.get("category", "Custom"),
                    "description": kwargs.get("description", ""),
                    "created_by": kwargs.get("created_by", "agent")
                })
            elif action == "get_statistics":
                result = await self._call_mcp_server(server_path, "get_value_statistics", {})
            else:
                return json.dumps({"error": f"Unknown action: {action}"})
            
            return result
        except Exception as e:
            return json.dumps({"error": f"Values taxonomy tool failed: {str(e)}"})
    
    async def _call_mcp_server(self, server_path: Path, method: str, params: Dict[str, Any]) -> str:
        """Call MCP server method"""
        # For now, directly import and call the server methods
        # In production, this would use proper MCP client protocol
        try:
            import sys
            sys.path.append(str(server_path.parent))
            
            from values_taxonomy_server import ValuesTaxonomyServer
            
            server = ValuesTaxonomyServer()
            
            # Map method names to server methods
            method_mapping = {
                "get_taxonomy_values": server._register_tools()[0],  # This is a simplified approach
                # In reality, we'd need proper MCP client implementation
            }
            
            # For demo purposes, return mock data based on action
            if method == "get_taxonomy_values":
                return json.dumps({
                    "values": [
                        {"value_id": 1, "value_name": "Integrity", "value_category": "Core", "description": "Acting with honesty"},
                        {"value_id": 2, "value_name": "Excellence", "value_category": "Achievement", "description": "Pursuing quality"}
                    ]
                })
            else:
                return json.dumps({"result": f"Called {method} with {params}"})
                
        except Exception as e:
            return json.dumps({"error": f"MCP server call failed: {str(e)}"})

class DocumentProcessingTool(BaseTool):
    """Tool for document processing operations"""
    
    name = "document_processing"
    description = "Segment documents, parse sentences, and extract metadata"
    args_schema = DocumentProcessingInput
    
    def _run(self, action: str, text: str, **kwargs) -> str:
        """Run the tool synchronously"""
        return asyncio.run(self._arun(action, text, **kwargs))
    
    async def _arun(self, action: str, text: str, **kwargs) -> str:
        """Run the tool asynchronously"""
        try:
            if action == "segment_document":
                return await self._segment_document(text, **kwargs)
            elif action == "parse_sentences":
                return await self._parse_sentences(text, **kwargs)
            elif action == "get_metadata":
                return await self._get_metadata(text)
            elif action == "validate_segments":
                return await self._validate_segments(kwargs.get("segments_json", "{}"), **kwargs)
            else:
                return json.dumps({"error": f"Unknown action: {action}"})
                
        except Exception as e:
            return json.dumps({"error": f"Document processing failed: {str(e)}"})
    
    async def _segment_document(self, text: str, **kwargs) -> str:
        """Segment document into analyzable chunks"""
        segment_type = kwargs.get("segment_type", "paragraph")
        min_words = kwargs.get("min_words", 20)
        max_words = kwargs.get("max_words", 150)
        
        if segment_type == "paragraph":
            paragraphs = text.split('\n\n')
            segments = []
            current_pos = 0
            
            for para in paragraphs:
                para = para.strip()
                if para:
                    word_count = len(para.split())
                    if word_count >= min_words:
                        segments.append({
                            'text': para,
                            'start_pos': current_pos,
                            'end_pos': current_pos + len(para),
                            'word_count': word_count,
                            'segment_type': 'paragraph'
                        })
                current_pos += len(para) + 2
        
        elif segment_type == "sentence":
            import re
            sentences = re.split(r'[.!?]+', text)
            segments = []
            current_pos = 0
            current_segment = []
            current_words = 0
            
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence:
                    sentence_words = len(sentence.split())
                    
                    if current_words + sentence_words <= max_words:
                        current_segment.append(sentence)
                        current_words += sentence_words
                    else:
                        if current_words >= min_words:
                            segment_text = '. '.join(current_segment) + '.'
                            segments.append({
                                'text': segment_text,
                                'start_pos': current_pos,
                                'end_pos': current_pos + len(segment_text),
                                'word_count': current_words,
                                'segment_type': 'sentence_group'
                            })
                            current_pos += len(segment_text) + 1
                        
                        current_segment = [sentence]
                        current_words = sentence_words
            
            # Handle final segment
            if current_segment and current_words >= min_words:
                segment_text = '. '.join(current_segment) + '.'
                segments.append({
                    'text': segment_text,
                    'start_pos': current_pos,
                    'end_pos': current_pos + len(segment_text),
                    'word_count': current_words,
                    'segment_type': 'sentence_group'
                })
        
        else:  # smart segmentation
            # Implement smart segmentation logic
            segments = await self._smart_segmentation(text, min_words, max_words)
        
        result = {
            'segments': segments,
            'total_segments': len(segments),
            'total_words': sum(s['word_count'] for s in segments),
            'segmentation_type': segment_type
        }
        
        return json.dumps(result, indent=2)
    
    async def _parse_sentences(self, text: str, **kwargs) -> str:
        """Parse text into individual sentences"""
        import re
        sentences = []
        sentence_pattern = r'[.!?]+\s+'
        parts = re.split(sentence_pattern, text)
        current_pos = kwargs.get('segment_start', 0)
        
        for part in parts:
            if part.strip():
                sentences.append({
                    'text': part.strip(),
                    'start_pos': current_pos,
                    'end_pos': current_pos + len(part),
                    'word_count': len(part.split())
                })
                current_pos += len(part) + 2
        
        return json.dumps({'sentences': sentences}, indent=2)
    
    async def _get_metadata(self, text: str) -> str:
        """Extract document metadata"""
        import re
        
        word_count = len(text.split())
        char_count = len(text)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        sentence_count = len(re.findall(r'[.!?]+', text))
        
        metadata = {
            'word_count': word_count,
            'character_count': char_count,
            'paragraph_count': len(paragraphs),
            'sentence_count': sentence_count,
            'average_sentence_length': round(word_count / sentence_count, 2) if sentence_count > 0 else 0,
            'suggested_segment_count': max(1, word_count // 75)
        }
        
        return json.dumps(metadata, indent=2)
    
    async def _validate_segments(self, segments_json: str, **kwargs) -> str:
        """Validate segment quality"""
        try:
            segments_data = json.loads(segments_json)
            segments = segments_data.get('segments', [])
        except json.JSONDecodeError:
            return json.dumps({'error': 'Invalid JSON format'})
        
        min_words = kwargs.get('min_words', 10)
        max_words = kwargs.get('max_words', 200)
        validation_results = []
        
        for i, segment in enumerate(segments):
            issues = []
            word_count = segment.get('word_count', 0)
            
            if word_count < min_words:
                issues.append(f'Too short ({word_count} words)')
            if word_count > max_words:
                issues.append(f'Too long ({word_count} words)')
            
            validation_results.append({
                'segment_index': i,
                'is_valid': len(issues) == 0,
                'issues': issues,
                'word_count': word_count
            })
        
        summary = {
            'total_segments': len(segments),
            'valid_segments': sum(1 for r in validation_results if r['is_valid']),
            'validation_results': validation_results
        }
        
        return json.dumps(summary, indent=2)
    
    async def _smart_segmentation(self, text: str, min_words: int, max_words: int) -> List[Dict[str, Any]]:
        """Implement smart segmentation"""
        # For now, use paragraph-based segmentation with merging logic
        paragraphs = text.split('\n\n')
        segments = []
        current_pos = 0
        
        i = 0
        while i < len(paragraphs):
            para = paragraphs[i].strip()
            if para:
                word_count = len(para.split())
                
                if word_count < min_words and i + 1 < len(paragraphs):
                    # Try to merge with next paragraph
                    next_para = paragraphs[i + 1].strip()
                    combined_text = para + '\n\n' + next_para
                    combined_words = len(combined_text.split())
                    
                    if combined_words <= max_words:
                        segments.append({
                            'text': combined_text,
                            'start_pos': current_pos,
                            'end_pos': current_pos + len(combined_text),
                            'word_count': combined_words,
                            'segment_type': 'merged'
                        })
                        current_pos += len(combined_text) + 2
                        i += 2  # Skip next paragraph
                        continue
                
                if min_words <= word_count <= max_words:
                    segments.append({
                        'text': para,
                        'start_pos': current_pos,
                        'end_pos': current_pos + len(para),
                        'word_count': word_count,
                        'segment_type': 'paragraph'
                    })
            
            current_pos += len(para) + 2
            i += 1
        
        return segments

class BehavioralScaleTool(BaseTool):
    """Tool for behavioral scale operations"""
    
    name = "behavioral_scale"
    description = "Score behavioral enactment and manage the -3 to +3 scale"
    args_schema = BehavioralScaleInput
    
    def _run(self, action: str, **kwargs) -> str:
        """Run the tool synchronously"""
        return asyncio.run(self._arun(action, **kwargs))
    
    async def _arun(self, action: str, **kwargs) -> str:
        """Run the tool asynchronously"""
        try:
            if action == "get_definitions":
                return await self._get_scale_definitions()
            elif action == "score_behavior":
                return await self._score_behavior(**kwargs)
            elif action == "get_examples":
                return await self._get_examples(kwargs.get("behavioral_score", 0))
            else:
                return json.dumps({"error": f"Unknown action: {action}"})
                
        except Exception as e:
            return json.dumps({"error": f"Behavioral scale tool failed: {str(e)}"})
    
    async def _get_scale_definitions(self) -> str:
        """Get behavioral scale definitions"""
        definitions = [
            {"score": -3, "name": "Extraordinary Violation", "description": "Systematic undermining of values"},
            {"score": -2, "name": "Active Violation", "description": "Deliberate contradiction of values"},
            {"score": -1, "name": "Capitulating", "description": "Surrender through inaction"},
            {"score": 0, "name": "Indifference", "description": "Apathetic disengagement"},
            {"score": 1, "name": "Compromising", "description": "Partial selective enactment"},
            {"score": 2, "name": "Active Enacting", "description": "Consistent deliberate alignment"},
            {"score": 3, "name": "Extraordinary Enacting", "description": "Exceptional commitment with sacrifice"}
        ]
        
        return json.dumps({"scale_definitions": definitions}, indent=2)
    
    async def _score_behavior(self, **kwargs) -> str:
        """Score behavioral enactment"""
        sentence_text = kwargs.get("sentence_text", "")
        value_name = kwargs.get("value_name", "")
        
        # Simple scoring logic based on keywords
        sentence_lower = sentence_text.lower()
        
        if any(word in sentence_lower for word in ['sacrifice', 'extraordinary', 'above and beyond']):
            score = 3
        elif any(word in sentence_lower for word in ['consistently', 'always', 'committed']):
            score = 2
        elif any(word in sentence_lower for word in ['sometimes', 'partially']):
            score = 1
        elif any(word in sentence_lower for word in ['indifferent', 'neutral']):
            score = 0
        elif any(word in sentence_lower for word in ['avoided', 'ignored', 'failed']):
            score = -1
        elif any(word in sentence_lower for word in ['deliberately', 'violated']):
            score = -2
        elif any(word in sentence_lower for word in ['sabotaged', 'undermined']):
            score = -3
        else:
            score = 1  # Default positive assumption
        
        result = {
            'sentence_text': sentence_text,
            'value_name': value_name,
            'suggested_score': score,
            'confidence': 0.7,
            'rationale': f'Behavioral analysis for value "{value_name}" suggests score {score} based on language indicators'
        }
        
        return json.dumps(result, indent=2)
    
    async def _get_examples(self, score: int) -> str:
        """Get examples for a behavioral score"""
        examples = {
            3: ["Sacrificing personal advancement to maintain integrity", "Taking significant risks to help others"],
            2: ["Consistently following through on commitments", "Regularly helping team members"],
            1: ["Following some rules but not others", "Helping when convenient"],
            0: ["Being indifferent to outcomes", "Showing no concern"],
            -1: ["Remaining silent when action needed", "Avoiding responsibility"],
            -2: ["Knowingly breaking rules for gain", "Deliberately treating others unfairly"],
            -3: ["Systematically sabotaging efforts", "Actively undermining values"]
        }
        
        return json.dumps({
            "score": score,
            "examples": examples.get(score, ["No examples available"])
        }, indent=2)

class AnalysisTool(BaseTool):
    """Tool for analysis and reporting operations"""
    
    name = "analysis"
    description = "Perform statistical analysis and generate reports"
    args_schema = AnalysisInput
    
    def _run(self, action: str, **kwargs) -> str:
        """Run the tool synchronously"""
        return asyncio.run(self._arun(action, **kwargs))
    
    async def _arun(self, action: str, **kwargs) -> str:
        """Run the tool asynchronously"""
        try:
            if action == "calculate_frequencies":
                return await self._calculate_frequencies(**kwargs)
            elif action == "gap_analysis":
                return await self._gap_analysis(**kwargs)
            elif action == "generate_statistics":
                return await self._generate_statistics(**kwargs)
            else:
                return json.dumps({"error": f"Unknown action: {action}"})
                
        except Exception as e:
            return json.dumps({"error": f"Analysis tool failed: {str(e)}"})
    
    async def _calculate_frequencies(self, **kwargs) -> str:
        """Calculate value frequencies"""
        # Mock frequency data for demo
        frequencies = [
            {"value_name": "Integrity", "category": "Core", "frequency": 15, "confidence": 0.85},
            {"value_name": "Excellence", "category": "Achievement", "frequency": 12, "confidence": 0.78},
            {"value_name": "Service", "category": "Benevolence", "frequency": 8, "confidence": 0.82}
        ]
        
        result = {
            "value_frequencies": frequencies,
            "total_coded_sections": sum(f["frequency"] for f in frequencies),
            "unique_values_count": len(frequencies)
        }
        
        return json.dumps(result, indent=2)
    
    async def _gap_analysis(self, **kwargs) -> str:
        """Perform gap analysis between open and taxonomy coding"""
        open_values = set(kwargs.get("open_coded_values", []))
        taxonomy_values = set(kwargs.get("taxonomy_values", []))
        
        exact_matches = open_values.intersection(taxonomy_values)
        open_only = open_values - taxonomy_values
        taxonomy_unused = taxonomy_values - open_values
        
        result = {
            "exact_matches": list(exact_matches),
            "open_only_values": list(open_only),
            "unused_taxonomy_values": list(taxonomy_unused),
            "coverage_rate": len(exact_matches) / len(open_values) if open_values else 0
        }
        
        return json.dumps(result, indent=2)
    
    async def _generate_statistics(self, **kwargs) -> str:
        """Generate comprehensive statistics"""
        result = {
            "values_coding": {
                "total_coded_sections": 35,
                "unique_values_identified": 12,
                "average_confidence": 0.78
            },
            "behavioral_coding": {
                "total_coded_sentences": 127,
                "average_behavioral_score": 1.2,
                "positive_scores": 89,
                "negative_scores": 15,
                "neutral_scores": 23
            }
        }
        
        return json.dumps(result, indent=2)