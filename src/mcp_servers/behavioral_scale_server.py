"""
Behavioral Scale MCP Server
Handles the -3 to +3 behavioral enactment scale operations
"""

import json
import sqlite3
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

try:
    from mcp.server import Server
    from mcp.types import Tool, TextContent
    import mcp.server.stdio
except ImportError:
    print("MCP not available - install with: pip install mcp")
    Server = None
    Tool = None
    TextContent = None

@dataclass
class BehavioralScalePoint:
    score: int
    name: str
    short_description: str
    full_description: str
    examples: List[str]

class BehavioralIndicator(Enum):
    EXTRAORDINARY_VIOLATION = -3
    ACTIVE_VIOLATION = -2
    CAPITULATING = -1
    INDIFFERENCE = 0
    COMPROMISING = 1
    ACTIVE_ENACTING = 2
    EXTRAORDINARY_ENACTING = 3

class BehavioralScaleServer:
    def __init__(self, database_path: str = None):
        self.database_path = database_path or "values_behavioral.db"
        self.server = Server("behavioral-scale") if Server else None
        
        # Initialize scale definitions
        self.scale_definitions = self._init_scale_definitions()
        self._init_database()
        
        if self.server:
            self._register_tools()
    
    def _init_scale_definitions(self) -> Dict[int, BehavioralScalePoint]:
        """Initialize the behavioral scale definitions"""
        return {
            -3: BehavioralScalePoint(
                score=-3,
                name="Extraordinary Violation",
                short_description="Systematic undermining of values",
                full_description="Deliberately and systematically acting in ways that undermine or contradict the core value, often with significant negative consequences",
                examples=[
                    "Deliberately sabotaging team efforts",
                    "Systematically violating ethical standards",
                    "Actively working against organizational values"
                ]
            ),
            -2: BehavioralScalePoint(
                score=-2,
                name="Active Violation",
                short_description="Deliberate contradiction of values",
                full_description="Consciously and deliberately acting in ways that directly contradict or oppose the core value",
                examples=[
                    "Knowingly breaking rules for personal gain",
                    "Deliberately treating others unfairly",
                    "Actively resisting positive changes"
                ]
            ),
            -1: BehavioralScalePoint(
                score=-1,
                name="Capitulating",
                short_description="Surrender through inaction",
                full_description="Failing to act in alignment with values when action was possible, essentially surrendering to opposing forces through passivity",
                examples=[
                    "Remaining silent when speaking up is needed",
                    "Avoiding responsibility when leadership is required",
                    "Giving up on important principles under pressure"
                ]
            ),
            0: BehavioralScalePoint(
                score=0,
                name="Indifference",
                short_description="Apathetic disengagement",
                full_description="Showing no particular care or concern for the value, neither supporting nor opposing it, remaining neutral or disengaged",
                examples=[
                    "Being indifferent to team success or failure",
                    "Showing no concern for ethical issues",
                    "Remaining uninvolved in important decisions"
                ]
            ),
            1: BehavioralScalePoint(
                score=1,
                name="Compromising",
                short_description="Partial selective enactment",
                full_description="Acting in alignment with the value only partially or selectively, making compromises that weaken full commitment",
                examples=[
                    "Following some rules but not others",
                    "Helping some people but not others",
                    "Pursuing quality only when convenient"
                ]
            ),
            2: BehavioralScalePoint(
                score=2,
                name="Active Enacting",
                short_description="Consistent deliberate alignment",
                full_description="Consistently and deliberately acting in ways that align with and support the core value, making it a priority in decisions and actions",
                examples=[
                    "Consistently following through on commitments",
                    "Regularly helping team members",
                    "Maintaining high standards even under pressure"
                ]
            ),
            3: BehavioralScalePoint(
                score=3,
                name="Extraordinary Enacting",
                short_description="Exceptional commitment with sacrifice",
                full_description="Going above and beyond to enact the value, often at significant personal or professional cost, demonstrating exceptional commitment",
                examples=[
                    "Sacrificing personal advancement to maintain integrity",
                    "Taking significant risks to help others",
                    "Persevering through extreme challenges to uphold principles"
                ]
            )
        }
    
    def _init_database(self):
        """Initialize database with scale definitions"""
        conn = sqlite3.connect(self.database_path)
        try:
            cursor = conn.cursor()
            
            # Create behavioral scale table if not exists
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS behavioral_scale_definitions (
                    scale_point INTEGER PRIMARY KEY CHECK (scale_point BETWEEN -3 AND 3),
                    scale_name TEXT NOT NULL,
                    short_description TEXT NOT NULL,
                    full_description TEXT NOT NULL,
                    examples TEXT,
                    created_date TEXT DEFAULT (datetime('now'))
                )
            """)
            
            # Check if definitions exist, if not populate them
            cursor.execute("SELECT COUNT(*) FROM behavioral_scale_definitions")
            if cursor.fetchone()[0] == 0:
                for score, definition in self.scale_definitions.items():
                    cursor.execute("""
                        INSERT INTO behavioral_scale_definitions 
                        (scale_point, scale_name, short_description, full_description, examples)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        definition.score,
                        definition.name,
                        definition.short_description,
                        definition.full_description,
                        json.dumps(definition.examples)
                    ))
                conn.commit()
            
            # Create behavioral scoring history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS behavioral_scoring_history (
                    history_id INTEGER PRIMARY KEY,
                    sentence_text TEXT NOT NULL,
                    value_name TEXT NOT NULL,
                    behavioral_score INTEGER NOT NULL CHECK (behavioral_score BETWEEN -3 AND 3),
                    confidence_score REAL CHECK (confidence_score BETWEEN 0.0 AND 1.0),
                    rationale TEXT,
                    context TEXT,
                    scored_by TEXT,
                    scored_date TEXT DEFAULT (datetime('now')),
                    model_version TEXT
                )
            """)
            conn.commit()
        finally:
            conn.close()
    
    def _register_tools(self):
        """Register MCP tools"""
        
        @self.server.call_tool()
        async def get_scale_definitions() -> List[TextContent]:
            """Get all behavioral scale definitions"""
            definitions = []
            for score, definition in self.scale_definitions.items():
                definitions.append({
                    'score': definition.score,
                    'name': definition.name,
                    'short_description': definition.short_description,
                    'full_description': definition.full_description,
                    'examples': definition.examples
                })
            
            return [TextContent(
                type="text",
                text=json.dumps({"scale_definitions": definitions}, indent=2)
            )]
        
        @self.server.call_tool()
        async def score_behavior(
            sentence_text: str, 
            value_name: str, 
            context: str = "", 
            scored_by: str = "ai",
            model_version: str = "claude-3.5-sonnet"
        ) -> List[TextContent]:
            """Analyze sentence and suggest behavioral score"""
            
            # Analyze the sentence for behavioral indicators
            analysis = self._analyze_behavioral_content(sentence_text, value_name, context)
            
            # Store in history
            conn = sqlite3.connect(self.database_path)
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO behavioral_scoring_history 
                    (sentence_text, value_name, behavioral_score, confidence_score, rationale, context, scored_by, model_version)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    sentence_text, value_name, analysis['suggested_score'],
                    analysis['confidence'], analysis['rationale'], context, scored_by, model_version
                ))
                history_id = cursor.lastrowid
                conn.commit()
            finally:
                conn.close()
            
            result = {
                'sentence_text': sentence_text,
                'value_name': value_name,
                'suggested_score': analysis['suggested_score'],
                'confidence': analysis['confidence'],
                'rationale': analysis['rationale'],
                'scale_point_info': self.scale_definitions[analysis['suggested_score']].__dict__,
                'history_id': history_id,
                'indicators_found': analysis['indicators'],
                'context_factors': analysis['context_factors']
            }
            
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2, default=str)
            )]
        
        @self.server.call_tool()
        async def get_scale_examples(score: int) -> List[TextContent]:
            """Get examples for a specific scale point"""
            if score not in self.scale_definitions:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": f"Invalid score: {score}. Must be between -3 and 3"})
                )]
            
            definition = self.scale_definitions[score]
            result = {
                'score': score,
                'name': definition.name,
                'description': definition.full_description,
                'examples': definition.examples
            }
            
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
        
        @self.server.call_tool()
        async def validate_scoring(proposed_score: int, sentence_text: str, value_name: str, rationale: str) -> List[TextContent]:
            """Validate a proposed behavioral score"""
            validation_results = []
            
            # Check score bounds
            if not -3 <= proposed_score <= 3:
                validation_results.append({
                    'type': 'error',
                    'message': f'Score {proposed_score} is out of bounds. Must be between -3 and 3.'
                })
                return [TextContent(type="text", text=json.dumps({"validation": validation_results}))]
            
            # Analyze sentence for contradictory indicators
            analysis = self._analyze_behavioral_content(sentence_text, value_name, "")
            suggested_score = analysis['suggested_score']
            
            # Check for major discrepancies
            score_difference = abs(proposed_score - suggested_score)
            if score_difference > 2:
                validation_results.append({
                    'type': 'warning',
                    'message': f'Large discrepancy: Proposed score {proposed_score}, AI suggested {suggested_score}',
                    'suggestion': 'Consider reviewing the rationale and sentence content'
                })
            elif score_difference > 1:
                validation_results.append({
                    'type': 'caution',
                    'message': f'Moderate discrepancy: Proposed {proposed_score} vs AI suggested {suggested_score}',
                    'suggestion': 'Double-check behavioral indicators in the text'
                })
            
            # Validate rationale quality
            if len(rationale.strip()) < 10:
                validation_results.append({
                    'type': 'warning',
                    'message': 'Rationale is very short',
                    'suggestion': 'Consider providing more detailed justification'
                })
            
            # Check for consistency with scale definition
            scale_def = self.scale_definitions[proposed_score]
            keywords = self._extract_behavioral_keywords(sentence_text)
            
            result = {
                'is_valid': len([r for r in validation_results if r['type'] == 'error']) == 0,
                'validation_issues': validation_results,
                'ai_suggested_score': suggested_score,
                'ai_rationale': analysis['rationale'],
                'scale_definition': {
                    'name': scale_def.name,
                    'description': scale_def.short_description
                },
                'behavioral_keywords_found': keywords
            }
            
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
        
        @self.server.call_tool()
        async def calculate_reliability(scorer_a_scores: List[int], scorer_b_scores: List[int]) -> List[TextContent]:
            """Calculate inter-rater reliability statistics"""
            if len(scorer_a_scores) != len(scorer_b_scores):
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": "Score arrays must be the same length"})
                )]
            
            n = len(scorer_a_scores)
            if n == 0:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": "No scores provided"})
                )]
            
            # Calculate agreement statistics
            exact_agreements = sum(1 for a, b in zip(scorer_a_scores, scorer_b_scores) if a == b)
            close_agreements = sum(1 for a, b in zip(scorer_a_scores, scorer_b_scores) if abs(a - b) <= 1)
            
            # Correlation coefficient
            mean_a = sum(scorer_a_scores) / n
            mean_b = sum(scorer_b_scores) / n
            
            numerator = sum((a - mean_a) * (b - mean_b) for a, b in zip(scorer_a_scores, scorer_b_scores))
            denom_a = sum((a - mean_a) ** 2 for a in scorer_a_scores)
            denom_b = sum((b - mean_b) ** 2 for b in scorer_b_scores)
            
            correlation = numerator / (denom_a * denom_b) ** 0.5 if denom_a * denom_b > 0 else 0
            
            # Mean absolute difference
            mean_abs_diff = sum(abs(a - b) for a, b in zip(scorer_a_scores, scorer_b_scores)) / n
            
            result = {
                'sample_size': n,
                'exact_agreement_rate': exact_agreements / n,
                'close_agreement_rate': close_agreements / n,  # Within 1 point
                'correlation_coefficient': round(correlation, 3),
                'mean_absolute_difference': round(mean_abs_diff, 2),
                'scorer_a_mean': round(mean_a, 2),
                'scorer_b_mean': round(mean_b, 2),
                'reliability_assessment': self._assess_reliability(exact_agreements / n, correlation)
            }
            
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
    
    def _analyze_behavioral_content(self, sentence: str, value_name: str, context: str = "") -> Dict[str, Any]:
        """Analyze sentence content for behavioral indicators"""
        sentence_lower = sentence.lower()
        
        # Define behavioral indicator patterns
        positive_indicators = {
            3: ['sacrifice', 'extraordinary', 'despite', 'above and beyond', 'personal cost', 'risking'],
            2: ['consistently', 'always', 'committed', 'dedicated', 'actively', 'deliberately'],
            1: ['sometimes', 'partially', 'when convenient', 'selective', 'compromise']
        }
        
        negative_indicators = {
            -1: ['avoided', 'ignored', 'failed to', 'gave up', 'remained silent', 'passive'],
            -2: ['deliberately', 'knowingly', 'intentionally', 'actively opposed', 'violated'],
            -3: ['systematically', 'sabotaged', 'undermined', 'destroyed', 'betrayed']
        }
        
        # Neutral indicators
        neutral_indicators = ['indifferent', 'neutral', 'no concern', 'uninvolved', 'disengaged']
        
        # Score the sentence
        scores_found = []
        indicators_found = []
        
        # Check positive indicators
        for score, patterns in positive_indicators.items():
            for pattern in patterns:
                if pattern in sentence_lower:
                    scores_found.append(score)
                    indicators_found.append(f"Positive indicator '{pattern}' suggests score {score}")
        
        # Check negative indicators  
        for score, patterns in negative_indicators.items():
            for pattern in patterns:
                if pattern in sentence_lower:
                    scores_found.append(score)
                    indicators_found.append(f"Negative indicator '{pattern}' suggests score {score}")
        
        # Check neutral indicators
        for pattern in neutral_indicators:
            if pattern in sentence_lower:
                scores_found.append(0)
                indicators_found.append(f"Neutral indicator '{pattern}' suggests score 0")
        
        # Determine suggested score
        if scores_found:
            suggested_score = round(sum(scores_found) / len(scores_found))
        else:
            # Default scoring based on general sentiment
            if any(word in sentence_lower for word in ['help', 'support', 'assist', 'contribute']):
                suggested_score = 2
            elif any(word in sentence_lower for word in ['refuse', 'reject', 'oppose', 'against']):
                suggested_score = -1
            else:
                suggested_score = 1  # Neutral positive assumption
        
        # Ensure score is within bounds
        suggested_score = max(-3, min(3, suggested_score))
        
        # Calculate confidence based on number of indicators
        confidence = min(0.9, 0.3 + (len(indicators_found) * 0.15))
        
        # Generate rationale
        if indicators_found:
            rationale = f"Behavioral analysis for value '{value_name}': " + "; ".join(indicators_found[:3])
        else:
            rationale = f"General behavioral assessment for value '{value_name}' based on sentence content and typical behavioral patterns"
        
        # Context factors
        context_factors = []
        if 'team' in sentence_lower or 'others' in sentence_lower:
            context_factors.append('Social/collaborative context')
        if 'pressure' in sentence_lower or 'difficult' in sentence_lower:
            context_factors.append('Challenging circumstances')
        if 'decision' in sentence_lower or 'choice' in sentence_lower:
            context_factors.append('Decision-making context')
        
        return {
            'suggested_score': suggested_score,
            'confidence': confidence,
            'rationale': rationale,
            'indicators': indicators_found,
            'context_factors': context_factors
        }
    
    def _extract_behavioral_keywords(self, text: str) -> List[str]:
        """Extract behavioral keywords from text"""
        behavioral_keywords = [
            'action', 'behavior', 'decision', 'choice', 'commitment', 'dedication',
            'sacrifice', 'effort', 'priority', 'value', 'principle', 'standard',
            'help', 'support', 'oppose', 'resist', 'maintain', 'uphold', 'violate'
        ]
        
        text_lower = text.lower()
        found_keywords = [kw for kw in behavioral_keywords if kw in text_lower]
        return found_keywords
    
    def _assess_reliability(self, agreement_rate: float, correlation: float) -> str:
        """Assess reliability based on statistics"""
        if agreement_rate >= 0.8 and correlation >= 0.8:
            return "Excellent reliability"
        elif agreement_rate >= 0.6 and correlation >= 0.6:
            return "Good reliability"
        elif agreement_rate >= 0.4 and correlation >= 0.4:
            return "Moderate reliability"
        else:
            return "Poor reliability - consider additional training"

async def main():
    """Run the Behavioral Scale MCP Server"""
    if not Server:
        print("MCP library not available. Install with: pip install mcp")
        return
    
    server = BehavioralScaleServer()
    
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.server.run(
            read_stream,
            write_stream,
            server.server.create_initialization_options()
        )

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())