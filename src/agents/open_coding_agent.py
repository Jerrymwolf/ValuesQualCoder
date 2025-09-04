"""
Open Coding Agent
Performs unrestricted value identification without taxonomic constraints
"""

from typing import Dict, Any, List
from langchain.prompts import ChatPromptTemplate
from langchain.tools import BaseTool, tool
from .base_agent import BaseValuesAgent, AgentResult, validate_text_input, extract_json_from_response
from .mcp_tools import DocumentProcessingTool, AnalysisTool
import json
import re

class OpenCodingAgent(BaseValuesAgent):
    """
    Open Coding Agent that identifies values without taxonomic constraints.
    
    This agent performs the first stage of dual coding methodology:
    1. Analyzes text segments for value expressions
    2. Identifies values based on broad understanding (not constrained to predefined taxonomy)
    3. Provides confidence ratings and rationale for each identified value
    4. Groups values into suggested categories
    5. Identifies patterns and themes across segments
    """
    
    def __init__(self, **kwargs):
        super().__init__(name="open_coding", **kwargs)
        self.identified_values = []  # Track values across segments
        self.value_patterns = {}     # Track recurring patterns
    
    def _initialize_tools(self) -> List[BaseTool]:
        """Initialize open coding specific tools"""
        return [
            self._create_value_identifier_tool(),
            self._create_pattern_analyzer_tool(),
            self._create_confidence_assessor_tool(),
            DocumentProcessingTool(),
            AnalysisTool()
        ]
    
    def _create_prompt(self) -> ChatPromptTemplate:
        """Create open coding agent prompt"""
        return ChatPromptTemplate.from_messages([
            ("system", """You are the Open Coding Agent for values identification in behavioral event interviews.

Your role is to perform open coding - identifying values without being constrained by any predefined taxonomy. You have complete freedom to identify ANY values that emerge from the text.

CORE PRINCIPLES:
1. **Unrestricted Analysis**: Identify values based on their natural expression in the text, not limited to any predefined list
2. **Inductive Reasoning**: Let the data speak - identify what values actually appear rather than what you expect to find
3. **Contextual Understanding**: Consider the full context of how values are expressed, including implicit and explicit mentions
4. **Multiple Values**: A single text segment may express multiple values - identify all that are present
5. **Emergent Categories**: Group identified values into logical categories that emerge from the data

IDENTIFICATION CRITERIA:
- Look for explicit value statements ("I believe in...", "What matters to me is...")
- Identify implicit values demonstrated through actions and decisions
- Recognize values expressed through emotional language and priorities
- Consider values shown through what people avoid or reject
- Notice values expressed through aspirations and goals

OUTPUT REQUIREMENTS:
For each text segment, provide:
- Identified values (use precise, descriptive names)
- Confidence level (0.0-1.0) for each value identification
- Detailed rationale explaining why this value is present
- Suggested category for each value (emergent, not predetermined)
- Context quotes supporting the identification
- Any patterns or themes you notice

IMPORTANT: 
- Do not limit yourself to any predefined value list
- Create new value names if existing terms don't capture what you observe
- Be specific - "helping others" is better than "benevolence" if that's what the text shows
- Consider cultural and individual variations in value expression
- Maintain high standards for evidence - confidence should reflect strength of evidence

Return all responses as structured JSON with clear rationale for each identification."""),
            ("human", "{input}")
        ])
    
    def _create_value_identifier_tool(self) -> BaseTool:
        """Tool for identifying values in text segments"""
        @tool
        def identify_values_in_segment(segment_text: str, segment_context: str = "") -> str:
            """Identify values expressed in a text segment using open coding methodology"""
            
            if not validate_text_input(segment_text, min_length=5):
                return json.dumps({"error": "Invalid or too short text segment"})
            
            # Analyze the text for value indicators
            values_found = []
            
            # Look for explicit value statements
            explicit_patterns = [
                r"I believe in ([^.!?]+)",
                r"important to me is ([^.!?]+)",
                r"I value ([^.!?]+)",
                r"what matters is ([^.!?]+)",
                r"I care about ([^.!?]+)",
                r"I prioritize ([^.!?]+)"
            ]
            
            for pattern in explicit_patterns:
                matches = re.finditer(pattern, segment_text, re.IGNORECASE)
                for match in matches:
                    value_text = match.group(1).strip()
                    values_found.append({
                        "value": self._extract_value_from_text(value_text),
                        "evidence": match.group(0),
                        "confidence": 0.9,
                        "rationale": f"Explicitly stated value: '{match.group(0)}'",
                        "type": "explicit"
                    })
            
            # Look for action-based value demonstrations
            values_found.extend(self._identify_action_based_values(segment_text))
            
            # Look for decision-based values
            values_found.extend(self._identify_decision_based_values(segment_text))
            
            # Look for emotional/priority indicators
            values_found.extend(self._identify_emotional_values(segment_text))
            
            # Categorize and refine values
            categorized_values = self._categorize_values(values_found)
            
            # Remove duplicates and merge similar values
            final_values = self._deduplicate_and_merge_values(categorized_values)
            
            result = {
                "segment_text": segment_text,
                "identified_values": final_values,
                "total_values_found": len(final_values),
                "analysis_confidence": self._calculate_overall_confidence(final_values),
                "emerging_patterns": self._identify_patterns_in_segment(segment_text, final_values),
                "segment_summary": self._summarize_value_profile(final_values)
            }
            
            # Update agent's tracking
            self.identified_values.extend(final_values)
            
            return json.dumps(result, indent=2)
        
        return identify_values_in_segment
    
    def _create_pattern_analyzer_tool(self) -> BaseTool:
        """Tool for analyzing patterns across segments"""
        @tool
        def analyze_value_patterns(segments_data: str) -> str:
            """Analyze patterns and themes across multiple coded segments"""
            
            try:
                segments = json.loads(segments_data)
            except json.JSONDecodeError:
                return json.dumps({"error": "Invalid JSON format for segments data"})
            
            all_values = []
            value_frequency = {}
            category_frequency = {}
            confidence_levels = []
            
            # Aggregate data from all segments
            for segment in segments.get("segments", []):
                for value in segment.get("identified_values", []):
                    all_values.append(value)
                    
                    value_name = value.get("value", "unknown")
                    category = value.get("category", "uncategorized")
                    confidence = value.get("confidence", 0.0)
                    
                    value_frequency[value_name] = value_frequency.get(value_name, 0) + 1
                    category_frequency[category] = category_frequency.get(category, 0) + 1
                    confidence_levels.append(confidence)
            
            # Identify patterns
            patterns = {
                "recurring_values": self._find_recurring_values(value_frequency),
                "dominant_categories": self._find_dominant_categories(category_frequency),
                "confidence_patterns": self._analyze_confidence_patterns(confidence_levels),
                "value_co_occurrence": self._find_value_co_occurrence(segments.get("segments", [])),
                "thematic_clusters": self._identify_thematic_clusters(all_values),
                "emergent_themes": self._identify_emergent_themes(all_values)
            }
            
            # Generate insights
            insights = {
                "total_unique_values": len(value_frequency),
                "most_frequent_value": max(value_frequency, key=value_frequency.get) if value_frequency else None,
                "average_confidence": sum(confidence_levels) / len(confidence_levels) if confidence_levels else 0,
                "category_distribution": category_frequency,
                "recommendations": self._generate_coding_recommendations(patterns)
            }
            
            result = {
                "pattern_analysis": patterns,
                "insights": insights,
                "value_frequencies": value_frequency,
                "analysis_metadata": {
                    "total_segments_analyzed": len(segments.get("segments", [])),
                    "total_values_identified": len(all_values),
                    "analysis_timestamp": self._get_timestamp()
                }
            }
            
            return json.dumps(result, indent=2)
        
        return analyze_value_patterns
    
    def _create_confidence_assessor_tool(self) -> BaseTool:
        """Tool for assessing and calibrating confidence levels"""
        @tool
        def assess_identification_confidence(value_data: str, segment_text: str) -> str:
            """Assess and potentially recalibrate confidence levels for value identifications"""
            
            try:
                value_info = json.loads(value_data)
            except json.JSONDecodeError:
                return json.dumps({"error": "Invalid JSON format for value data"})
            
            value_name = value_info.get("value", "")
            evidence = value_info.get("evidence", "")
            initial_confidence = value_info.get("confidence", 0.5)
            
            # Recalibrate confidence based on multiple factors
            confidence_factors = {
                "explicitness": self._assess_explicitness(evidence, segment_text),
                "context_strength": self._assess_context_strength(value_name, segment_text),
                "evidence_quality": self._assess_evidence_quality(evidence),
                "consistency": self._assess_consistency_with_segment(value_name, segment_text),
                "ambiguity": self._assess_ambiguity(value_name, evidence, segment_text)
            }
            
            # Calculate calibrated confidence
            calibrated_confidence = self._calibrate_confidence(initial_confidence, confidence_factors)
            
            # Generate confidence explanation
            explanation = self._generate_confidence_explanation(confidence_factors, calibrated_confidence)
            
            result = {
                "value": value_name,
                "initial_confidence": initial_confidence,
                "calibrated_confidence": round(calibrated_confidence, 3),
                "confidence_factors": confidence_factors,
                "explanation": explanation,
                "recommendation": self._get_confidence_recommendation(calibrated_confidence)
            }
            
            return json.dumps(result, indent=2)
        
        return assess_identification_confidence
    
    def _process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process open coding input"""
        required_fields = ['text_segments']
        
        for field in required_fields:
            if field not in input_data:
                raise ValueError(f"Missing required field: {field}")
        
        segments = input_data['text_segments']
        if not isinstance(segments, list) or len(segments) == 0:
            raise ValueError("text_segments must be a non-empty list")
        
        # Validate each segment
        for i, segment in enumerate(segments):
            if not isinstance(segment, dict) or 'text' not in segment:
                raise ValueError(f"Segment {i} must be a dict with 'text' field")
            
            if not validate_text_input(segment['text']):
                raise ValueError(f"Segment {i} has invalid text content")
        
        return {
            'text_segments': segments,
            'coding_mode': input_data.get('coding_mode', 'open'),
            'analysis_depth': input_data.get('analysis_depth', 'standard'),
            'include_patterns': input_data.get('include_patterns', True),
            'confidence_threshold': input_data.get('confidence_threshold', 0.3)
        }
    
    def _post_process_result(self, result: Dict[str, Any]) -> AgentResult:
        """Post-process open coding results"""
        try:
            output = result.get('output', '')
            
            # Extract structured data from response
            structured_result = extract_json_from_response(output)
            
            if not structured_result:
                # If no JSON found, try to parse the output differently
                structured_result = {
                    'raw_output': output,
                    'parsing_error': 'Could not extract structured JSON from response'
                }
            
            # Validate and enrich the result
            if 'identified_values' in structured_result:
                # Add summary statistics
                values = structured_result['identified_values']
                structured_result['summary'] = {
                    'total_values': len(values),
                    'high_confidence_values': len([v for v in values if v.get('confidence', 0) > 0.7]),
                    'categories_identified': len(set(v.get('category', 'unknown') for v in values)),
                    'average_confidence': sum(v.get('confidence', 0) for v in values) / len(values) if values else 0
                }
            
            return AgentResult(
                success=True,
                data=structured_result,
                metadata={
                    'agent_type': 'open_coding',
                    'values_identified_count': len(self.identified_values),
                    'analysis_method': 'unrestricted_open_coding'
                }
            )
        except Exception as e:
            return AgentResult(
                success=False,
                error=f"Open coding result processing failed: {str(e)}",
                data={'raw_result': result}
            )
    
    # Helper methods for value identification
    
    def _extract_value_from_text(self, text: str) -> str:
        """Extract a clean value name from matched text"""
        # Clean up the text to get a clear value name
        value_text = text.lower().strip()
        
        # Remove common connecting words
        connecting_words = ['the', 'a', 'an', 'is', 'being', 'to', 'of', 'for', 'with', 'in', 'on', 'at']
        words = value_text.split()
        cleaned_words = [w for w in words if w not in connecting_words]
        
        # Create a meaningful value name
        if cleaned_words:
            # Capitalize and join key words
            value_name = ' '.join(word.capitalize() for word in cleaned_words[:3])  # Limit to 3 words
        else:
            value_name = text.strip()
        
        return value_name
    
    def _identify_action_based_values(self, text: str) -> List[Dict[str, Any]]:
        """Identify values demonstrated through actions"""
        values = []
        
        # Action patterns that suggest values
        action_patterns = {
            r"I (decided|chose|opted) to ([^.!?]+)": ("Decision-Making", 0.7),
            r"I (helped|assisted|supported) ([^.!?]+)": ("Helping Others", 0.8),
            r"I (stood up|fought|advocated) for ([^.!?]+)": ("Advocacy", 0.8),
            r"I (worked|strived|pushed) to ([^.!?]+)": ("Achievement Orientation", 0.7),
            r"I (refused|rejected|declined) to ([^.!?]+)": ("Boundary Setting", 0.6),
            r"I (sacrificed|gave up) ([^.!?]+) (for|to) ([^.!?]+)": ("Self-Sacrifice", 0.9)
        }
        
        for pattern, (value_name, confidence) in action_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                values.append({
                    "value": value_name,
                    "evidence": match.group(0),
                    "confidence": confidence,
                    "rationale": f"Action-based value demonstrated: {match.group(0)}",
                    "type": "action-based"
                })
        
        return values
    
    def _identify_decision_based_values(self, text: str) -> List[Dict[str, Any]]:
        """Identify values shown through decision-making"""
        values = []
        
        decision_indicators = [
            r"I had to (choose|decide|pick) between ([^.!?]+)",
            r"The (difficult|hard|tough) decision was ([^.!?]+)",
            r"I (weighed|considered|thought about) ([^.!?]+)",
            r"What influenced my decision was ([^.!?]+)",
            r"I couldn't (compromise|give up) ([^.!?]+)"
        ]
        
        for pattern in decision_indicators:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Extract the decision context and infer values
                context = match.group(0)
                inferred_value = self._infer_value_from_decision_context(context)
                
                if inferred_value:
                    values.append({
                        "value": inferred_value,
                        "evidence": context,
                        "confidence": 0.6,
                        "rationale": f"Value inferred from decision context: {context}",
                        "type": "decision-based"
                    })
        
        return values
    
    def _identify_emotional_values(self, text: str) -> List[Dict[str, Any]]:
        """Identify values expressed through emotional language"""
        values = []
        
        emotional_patterns = {
            r"I (love|enjoy|am passionate about) ([^.!?]+)": ("Personal Fulfillment", 0.7),
            r"I (hate|dislike|can't stand) ([^.!?]+)": ("Value Opposition", 0.6),
            r"I (worry|am concerned) about ([^.!?]+)": ("Care and Concern", 0.6),
            r"I (am proud|take pride) in ([^.!?]+)": ("Pride and Achievement", 0.8),
            r"I (feel strongly|am committed) to ([^.!?]+)": ("Strong Commitment", 0.8),
            r"It (bothers|frustrates|upsets) me when ([^.!?]+)": ("Justice Sensitivity", 0.7)
        }
        
        for pattern, (value_category, confidence) in emotional_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                object_of_emotion = match.group(2)
                specific_value = self._extract_value_from_emotional_object(object_of_emotion)
                
                values.append({
                    "value": specific_value,
                    "evidence": match.group(0),
                    "confidence": confidence,
                    "rationale": f"Emotional expression indicating value: {match.group(0)}",
                    "type": "emotional"
                })
        
        return values
    
    def _categorize_values(self, values: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Categorize identified values into emergent categories"""
        
        # Define category patterns (these emerge from data, not predetermined)
        category_patterns = {
            "Interpersonal": ["help", "support", "relationship", "team", "others", "family", "friend"],
            "Personal Growth": ["learn", "develop", "grow", "improve", "challenge", "skill"],
            "Integrity": ["honest", "truth", "moral", "ethical", "principle", "right"],
            "Achievement": ["success", "accomplish", "goal", "excel", "perform", "quality"],
            "Service": ["serve", "contribute", "give", "volunteer", "community", "society"],
            "Leadership": ["lead", "guide", "mentor", "influence", "direct", "manage"],
            "Innovation": ["create", "new", "change", "innovate", "creative", "original"],
            "Stability": ["security", "stable", "predictable", "safe", "consistent", "reliable"],
            "Freedom": ["independent", "autonomous", "choice", "freedom", "control", "own"]
        }
        
        for value in values:
            value_name = value.get("value", "").lower()
            evidence = value.get("evidence", "").lower()
            
            # Find best matching category
            best_category = "Emerging"
            max_matches = 0
            
            for category, keywords in category_patterns.items():
                matches = sum(1 for keyword in keywords if keyword in value_name or keyword in evidence)
                if matches > max_matches:
                    max_matches = matches
                    best_category = category
            
            value["category"] = best_category
            value["category_confidence"] = min(1.0, max_matches / 3.0) if max_matches > 0 else 0.1
        
        return values
    
    def _deduplicate_and_merge_values(self, values: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicates and merge similar values"""
        
        if not values:
            return []
        
        # Group similar values
        value_groups = {}
        
        for value in values:
            value_name = value.get("value", "").lower()
            
            # Find if this value is similar to existing ones
            merged = False
            for existing_name, group in value_groups.items():
                if self._values_are_similar(value_name, existing_name):
                    group.append(value)
                    merged = True
                    break
            
            if not merged:
                value_groups[value_name] = [value]
        
        # Merge each group into a single value
        merged_values = []
        for group in value_groups.values():
            if len(group) == 1:
                merged_values.append(group[0])
            else:
                merged_value = self._merge_value_group(group)
                merged_values.append(merged_value)
        
        # Sort by confidence
        merged_values.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        
        return merged_values
    
    def _values_are_similar(self, value1: str, value2: str, threshold: float = 0.7) -> bool:
        """Check if two value names are similar enough to merge"""
        from difflib import SequenceMatcher
        
        similarity = SequenceMatcher(None, value1, value2).ratio()
        return similarity >= threshold
    
    def _merge_value_group(self, value_group: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge a group of similar values into one"""
        
        # Use the value with highest confidence as base
        base_value = max(value_group, key=lambda x: x.get("confidence", 0))
        
        # Combine evidence from all values
        all_evidence = []
        all_rationales = []
        confidence_sum = 0
        
        for value in value_group:
            all_evidence.append(value.get("evidence", ""))
            all_rationales.append(value.get("rationale", ""))
            confidence_sum += value.get("confidence", 0)
        
        # Create merged value
        merged = base_value.copy()
        merged["evidence"] = " | ".join(filter(None, all_evidence))
        merged["rationale"] = "Merged evidence: " + " | ".join(filter(None, all_rationales))
        merged["confidence"] = min(1.0, confidence_sum / len(value_group))  # Average confidence
        merged["merged_from"] = len(value_group)
        
        return merged
    
    def _calculate_overall_confidence(self, values: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence for the segment analysis"""
        if not values:
            return 0.0
        
        confidences = [v.get("confidence", 0) for v in values]
        return sum(confidences) / len(confidences)
    
    def _identify_patterns_in_segment(self, text: str, values: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify patterns within a single segment"""
        
        patterns = {
            "value_density": len(values) / max(1, len(text.split())),  # Values per word
            "dominant_types": self._find_dominant_value_types(values),
            "co_occurring_values": self._find_co_occurring_values_in_segment(values),
            "narrative_structure": self._analyze_narrative_structure(text, values)
        }
        
        return patterns
    
    def _summarize_value_profile(self, values: List[Dict[str, Any]]) -> str:
        """Create a summary of the value profile for this segment"""
        if not values:
            return "No clear values identified in this segment"
        
        categories = list(set(v.get("category", "unknown") for v in values))
        high_conf_values = [v for v in values if v.get("confidence", 0) > 0.7]
        
        summary_parts = []
        
        if high_conf_values:
            top_values = [v.get("value", "") for v in high_conf_values[:3]]
            summary_parts.append(f"Primary values: {', '.join(top_values)}")
        
        if categories:
            summary_parts.append(f"Categories: {', '.join(categories)}")
        
        confidence_avg = self._calculate_overall_confidence(values)
        summary_parts.append(f"Overall confidence: {confidence_avg:.2f}")
        
        return " | ".join(summary_parts)
    
    # Pattern analysis helper methods
    
    def _find_recurring_values(self, value_frequency: Dict[str, int]) -> List[Dict[str, Any]]:
        """Find values that appear multiple times"""
        recurring = []
        for value, freq in value_frequency.items():
            if freq > 1:
                recurring.append({"value": value, "frequency": freq})
        
        return sorted(recurring, key=lambda x: x["frequency"], reverse=True)
    
    def _find_dominant_categories(self, category_frequency: Dict[str, int]) -> List[Dict[str, Any]]:
        """Find most frequent value categories"""
        dominant = []
        total = sum(category_frequency.values())
        
        for category, freq in category_frequency.items():
            percentage = (freq / total) * 100 if total > 0 else 0
            dominant.append({
                "category": category,
                "frequency": freq,
                "percentage": round(percentage, 1)
            })
        
        return sorted(dominant, key=lambda x: x["frequency"], reverse=True)
    
    def _analyze_confidence_patterns(self, confidence_levels: List[float]) -> Dict[str, Any]:
        """Analyze confidence level patterns"""
        if not confidence_levels:
            return {}
        
        return {
            "average": sum(confidence_levels) / len(confidence_levels),
            "min": min(confidence_levels),
            "max": max(confidence_levels),
            "high_confidence_count": len([c for c in confidence_levels if c > 0.7]),
            "low_confidence_count": len([c for c in confidence_levels if c < 0.5])
        }
    
    def _find_value_co_occurrence(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find values that frequently appear together"""
        co_occurrences = {}
        
        for segment in segments:
            values_in_segment = [v.get("value", "") for v in segment.get("identified_values", [])]
            
            # Check all pairs
            for i in range(len(values_in_segment)):
                for j in range(i + 1, len(values_in_segment)):
                    pair = tuple(sorted([values_in_segment[i], values_in_segment[j]]))
                    co_occurrences[pair] = co_occurrences.get(pair, 0) + 1
        
        # Convert to list and sort
        co_occurrence_list = []
        for pair, count in co_occurrences.items():
            if count > 1:  # Only include pairs that occur more than once
                co_occurrence_list.append({
                    "values": list(pair),
                    "co_occurrence_count": count
                })
        
        return sorted(co_occurrence_list, key=lambda x: x["co_occurrence_count"], reverse=True)
    
    def _identify_thematic_clusters(self, all_values: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify thematic clusters of values"""
        # Group by category first
        category_groups = {}
        for value in all_values:
            category = value.get("category", "unknown")
            if category not in category_groups:
                category_groups[category] = []
            category_groups[category].append(value)
        
        clusters = []
        for category, values in category_groups.items():
            if len(values) >= 2:  # Only consider clusters with 2+ values
                cluster = {
                    "theme": category,
                    "values": [v.get("value", "") for v in values],
                    "cluster_size": len(values),
                    "average_confidence": sum(v.get("confidence", 0) for v in values) / len(values)
                }
                clusters.append(cluster)
        
        return sorted(clusters, key=lambda x: x["cluster_size"], reverse=True)
    
    def _identify_emergent_themes(self, all_values: List[Dict[str, Any]]) -> List[str]:
        """Identify emergent themes from the value set"""
        # This is a simplified version - in practice, this might use more sophisticated NLP
        themes = []
        
        # Look for common themes in rationales and evidence
        all_text = " ".join([
            v.get("rationale", "") + " " + v.get("evidence", "")
            for v in all_values
        ]).lower()
        
        theme_keywords = {
            "Leadership": ["lead", "manage", "direct", "guide", "mentor"],
            "Service": ["help", "serve", "support", "assist", "contribute"],
            "Growth": ["learn", "develop", "grow", "improve", "evolve"],
            "Relationships": ["team", "family", "friend", "colleague", "community"],
            "Quality": ["excellence", "quality", "standard", "best", "superior"]
        }
        
        for theme, keywords in theme_keywords.items():
            if sum(1 for keyword in keywords if keyword in all_text) >= 2:
                themes.append(theme)
        
        return themes
    
    def _generate_coding_recommendations(self, patterns: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on identified patterns"""
        recommendations = []
        
        recurring_values = patterns.get("recurring_values", [])
        if len(recurring_values) > 5:
            recommendations.append("Consider consolidating recurring values to avoid over-coding")
        
        confidence_patterns = patterns.get("confidence_patterns", {})
        low_confidence_count = confidence_patterns.get("low_confidence_count", 0)
        if low_confidence_count > 0:
            recommendations.append(f"Review {low_confidence_count} low-confidence value identifications")
        
        thematic_clusters = patterns.get("thematic_clusters", [])
        if len(thematic_clusters) < 2:
            recommendations.append("Consider if additional thematic categories are needed")
        
        return recommendations
    
    # Confidence assessment helper methods
    
    def _assess_explicitness(self, evidence: str, context: str) -> float:
        """Assess how explicitly the value is stated"""
        explicit_phrases = ["I believe", "I value", "important to me", "I care about"]
        explicitness_score = 0.3  # Base score
        
        for phrase in explicit_phrases:
            if phrase.lower() in evidence.lower():
                explicitness_score = 0.9
                break
        
        return explicitness_score
    
    def _assess_context_strength(self, value: str, context: str) -> float:
        """Assess how well the context supports the value identification"""
        # Simple keyword matching - in practice, this could be more sophisticated
        value_keywords = value.lower().split()
        context_lower = context.lower()
        
        matches = sum(1 for keyword in value_keywords if keyword in context_lower)
        return min(1.0, matches / len(value_keywords))
    
    def _assess_evidence_quality(self, evidence: str) -> float:
        """Assess the quality of evidence for value identification"""
        if len(evidence) < 10:
            return 0.2
        elif len(evidence) < 30:
            return 0.6
        else:
            return 0.8
    
    def _assess_consistency_with_segment(self, value: str, segment: str) -> float:
        """Assess consistency with overall segment theme"""
        # This would be more sophisticated in practice
        return 0.7  # Default moderate consistency
    
    def _assess_ambiguity(self, value: str, evidence: str, context: str) -> float:
        """Assess ambiguity level (lower is better)"""
        ambiguous_words = ["maybe", "might", "could", "possibly", "perhaps"]
        ambiguity_count = sum(1 for word in ambiguous_words if word in evidence.lower())
        
        return max(0.1, 1.0 - (ambiguity_count * 0.2))
    
    def _calibrate_confidence(self, initial_confidence: float, factors: Dict[str, float]) -> float:
        """Calibrate confidence based on multiple factors"""
        # Weighted combination of factors
        weights = {
            "explicitness": 0.3,
            "context_strength": 0.2,
            "evidence_quality": 0.2,
            "consistency": 0.15,
            "ambiguity": 0.15
        }
        
        weighted_score = sum(factors[factor] * weights[factor] for factor in factors)
        
        # Combine with initial confidence
        calibrated = (initial_confidence * 0.4) + (weighted_score * 0.6)
        
        return min(1.0, max(0.0, calibrated))
    
    def _generate_confidence_explanation(self, factors: Dict[str, float], final_confidence: float) -> str:
        """Generate explanation for confidence level"""
        explanations = []
        
        if factors.get("explicitness", 0) > 0.8:
            explanations.append("explicitly stated")
        elif factors.get("explicitness", 0) < 0.4:
            explanations.append("implicitly expressed")
        
        if factors.get("evidence_quality", 0) > 0.7:
            explanations.append("strong evidence")
        elif factors.get("evidence_quality", 0) < 0.4:
            explanations.append("limited evidence")
        
        if factors.get("ambiguity", 0) < 0.5:
            explanations.append("some ambiguity present")
        
        base_explanation = f"Confidence {final_confidence:.2f} based on: " + ", ".join(explanations)
        
        return base_explanation
    
    def _get_confidence_recommendation(self, confidence: float) -> str:
        """Get recommendation based on confidence level"""
        if confidence > 0.8:
            return "High confidence - include in analysis"
        elif confidence > 0.6:
            return "Moderate confidence - consider including with note"
        elif confidence > 0.4:
            return "Low confidence - review evidence or exclude"
        else:
            return "Very low confidence - likely false positive, exclude"
    
    # Additional helper methods
    
    def _infer_value_from_decision_context(self, context: str) -> str:
        """Infer value from decision-making context"""
        # Simple inference based on keywords in context
        context_lower = context.lower()
        
        if any(word in context_lower for word in ["team", "others", "colleague"]):
            return "Teamwork"
        elif any(word in context_lower for word in ["quality", "standard", "excellence"]):
            return "Quality Standards"
        elif any(word in context_lower for word in ["honest", "truth", "ethical"]):
            return "Honesty"
        elif any(word in context_lower for word in ["family", "personal", "balance"]):
            return "Work-Life Balance"
        else:
            return "Decision-Making Integrity"
    
    def _extract_value_from_emotional_object(self, emotional_object: str) -> str:
        """Extract specific value from object of emotional expression"""
        # Clean and extract meaningful value name
        cleaned = emotional_object.strip().lower()
        
        # Map common expressions to specific values
        value_mappings = {
            "team": "Teamwork",
            "quality": "Quality Standards",
            "people": "People Focus",
            "learning": "Learning and Development",
            "fairness": "Fairness",
            "innovation": "Innovation",
            "results": "Results Orientation"
        }
        
        for key, value in value_mappings.items():
            if key in cleaned:
                return value
        
        # Default: capitalize the object
        return emotional_object.strip().title()
    
    def _find_dominant_value_types(self, values: List[Dict[str, Any]]) -> Dict[str, int]:
        """Find dominant value types in a segment"""
        type_counts = {}
        for value in values:
            value_type = value.get("type", "unknown")
            type_counts[value_type] = type_counts.get(value_type, 0) + 1
        
        return type_counts
    
    def _find_co_occurring_values_in_segment(self, values: List[Dict[str, Any]]) -> List[List[str]]:
        """Find values that co-occur in a single segment"""
        if len(values) < 2:
            return []
        
        # Return all combinations of values in this segment
        value_names = [v.get("value", "") for v in values]
        return [value_names]  # Simplified - all values co-occur in this segment
    
    def _analyze_narrative_structure(self, text: str, values: List[Dict[str, Any]]) -> str:
        """Analyze the narrative structure of the segment"""
        # Simple analysis based on common narrative patterns
        text_lower = text.lower()
        
        if "when i" in text_lower or "i had to" in text_lower:
            return "situational_narrative"
        elif "i believe" in text_lower or "i value" in text_lower:
            return "declarative_values"
        elif "decided" in text_lower or "chose" in text_lower:
            return "decision_narrative"
        else:
            return "general_description"
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()