"""
Values and Behavioral Enactment Coder - Constants
Contains constants, enums, and configuration data for values and behavioral coding
"""

from enum import Enum
from typing import Dict, List, Tuple


class CodingStage(Enum):
    """Stages in the two-stage coding process"""
    VALUES = 1
    BEHAVIORAL = 2


class SessionStatus(Enum):
    """Status of coding sessions"""
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    PAUSED = "paused"


class ConfidenceLevel(Enum):
    """Confidence levels for coding decisions"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class SectionType(Enum):
    """Types of document sections"""
    PARAGRAPH = "paragraph"
    SENTENCE = "sentence"
    CUSTOM = "custom"


class ExportFormat(Enum):
    """Export formats for analysis"""
    CSV = "csv"
    JSON = "json"
    EXCEL = "excel"
    PDF = "pdf"


class CodingMode(Enum):
    """Modes for values coding"""
    OPEN = "open"  # Unrestricted value identification
    TAXONOMY = "taxonomy"  # Using predefined 32-value taxonomy
    HYBRID = "hybrid"  # Both approaches with comparison
    VALIDATION = "validation"  # Comparing open codes to taxonomy


class ExportType(Enum):
    """Types of export templates"""
    VALUES_SUMMARY = "values_summary"
    BEHAVIORAL_SUMMARY = "behavioral_summary"
    COMBINED = "combined"
    DETAILED = "detailed"
    OPEN_CODING_ANALYSIS = "open_coding_analysis"
    TAXONOMY_VALIDATION = "taxonomy_validation"


# Behavioral Scale Definitions
BEHAVIORAL_SCALE: Dict[int, Dict[str, str]] = {
    -3: {
        "name": "Extraordinary Violation",
        "short_description": "Systematic undermining of values",
        "full_description": "Deliberately and systematically acting in ways that undermine or contradict the core value, often with significant negative consequences",
        "examples": "Deliberately sabotaging team efforts, systematically violating ethical standards, actively working against organizational values",
        "color": "#FF0000"  # Red
    },
    -2: {
        "name": "Active Violation",
        "short_description": "Deliberate contradiction of values",
        "full_description": "Consciously and deliberately acting in ways that directly contradict or oppose the core value",
        "examples": "Knowingly breaking rules for personal gain, deliberately treating others unfairly, actively resisting positive changes",
        "color": "#FF6666"  # Light Red
    },
    -1: {
        "name": "Capitulating",
        "short_description": "Surrender through inaction",
        "full_description": "Failing to act in alignment with values when action was possible, essentially surrendering to opposing forces through passivity",
        "examples": "Remaining silent when speaking up is needed, avoiding responsibility when leadership is required, giving up on important principles under pressure",
        "color": "#FFAA99"  # Very Light Red
    },
    0: {
        "name": "Indifference",
        "short_description": "Apathetic disengagement",
        "full_description": "Showing no particular care or concern for the value, neither supporting nor opposing it, remaining neutral or disengaged",
        "examples": "Being indifferent to team success or failure, showing no concern for ethical issues, remaining uninvolved in important decisions",
        "color": "#CCCCCC"  # Gray
    },
    1: {
        "name": "Compromising",
        "short_description": "Partial selective enactment",
        "full_description": "Acting in alignment with the value only partially or selectively, making compromises that weaken full commitment",
        "examples": "Following some rules but not others, helping some people but not others, pursuing quality only when convenient",
        "color": "#AAFFAA"  # Very Light Green
    },
    2: {
        "name": "Active Enacting",
        "short_description": "Consistent deliberate alignment",
        "full_description": "Consistently and deliberately acting in ways that align with and support the core value, making it a priority in decisions and actions",
        "examples": "Consistently following through on commitments, regularly helping team members, maintaining high standards even under pressure",
        "color": "#66FF66"  # Light Green
    },
    3: {
        "name": "Extraordinary Enacting",
        "short_description": "Exceptional commitment with sacrifice",
        "full_description": "Going above and beyond to enact the value, often at significant personal or professional cost, demonstrating exceptional commitment",
        "examples": "Sacrificing personal advancement to maintain integrity, taking significant risks to help others, persevering through extreme challenges to uphold principles",
        "color": "#00FF00"  # Green
    }
}

# Core Values Taxonomy organized by category
CORE_VALUES_TAXONOMY: Dict[str, List[Tuple[str, str, str]]] = {
    "Achievement": [
        ("Achievement", "Personal accomplishment and success", "The drive to accomplish personal goals and demonstrate competence"),
        ("Excellence", "Pursuing the highest quality and standards", "Commitment to superior performance and continuous improvement"),
        ("Recognition", "Seeking acknowledgment and appreciation", "Desire for appreciation, respect, and acknowledgment from others"),
        ("Competence", "Developing and demonstrating skills", "Building and showcasing professional and personal capabilities"),
    ],
    "Benevolence": [
        ("Benevolence", "Concern for the welfare of others", "Acting with kindness and generosity toward others"),
        ("Service", "Helping and supporting others", "Dedication to meeting the needs of others and contributing to their wellbeing"),
        ("Compassion", "Empathy and care for others", "Deep concern for the suffering and needs of others"),
        ("Generosity", "Giving freely to others", "Willingness to share resources, time, and effort with others"),
    ],
    "Conformity": [
        ("Conformity", "Following rules and social expectations", "Adherence to social norms, rules, and expectations"),
        ("Tradition", "Respecting customs and established practices", "Commitment to maintaining cultural and organizational traditions"),
        ("Obedience", "Following authority and instructions", "Compliance with legitimate authority and established procedures"),
        ("Politeness", "Social courtesy and respect", "Maintaining proper social behavior and showing respect to others"),
    ],
    "Hedonism": [
        ("Hedonism", "Seeking pleasure and enjoyment", "Pursuit of personal pleasure, enjoyment, and gratification"),
        ("Fun", "Engaging in enjoyable activities", "Seeking entertainment, amusement, and lighthearted experiences"),
        ("Comfort", "Seeking ease and convenience", "Preference for comfortable, convenient, and stress-free conditions"),
    ],
    "Power": [
        ("Power", "Seeking control and influence", "Desire for control, dominance, and influence over others and situations"),
        ("Authority", "Exercising leadership and control", "Seeking positions of authority and decision-making responsibility"),
        ("Dominance", "Controlling others and situations", "Desire to control and direct others behavior and outcomes"),
        ("Status", "Seeking high social position", "Pursuit of prestige, social rank, and elevated social position"),
    ],
    "Security": [
        ("Security", "Seeking safety and stability", "Desire for safety, stability, and predictability in life and work"),
        ("Safety", "Protecting from harm and danger", "Ensuring protection from physical, emotional, and financial risks"),
        ("Stability", "Maintaining consistent conditions", "Preference for predictable, steady, and unchanging circumstances"),
        ("Order", "Organization and systematic approach", "Maintaining structure, organization, and systematic processes"),
    ],
    "Self-Direction": [
        ("Self-Direction", "Independence and autonomy", "Acting independently and making autonomous decisions"),
        ("Independence", "Freedom from external control", "Operating without external constraints or supervision"),
        ("Autonomy", "Self-governance and choice", "Having the freedom to make ones own choices and decisions"),
        ("Creativity", "Innovation and original thinking", "Developing new ideas, approaches, and creative solutions"),
    ],
    "Stimulation": [
        ("Stimulation", "Seeking excitement and novelty", "Pursuit of excitement, novelty, and challenging experiences"),
        ("Adventure", "Seeking new and exciting experiences", "Pursuing novel, thrilling, and adventurous activities"),
        ("Variety", "Seeking diverse experiences", "Preference for diverse, varied, and changing experiences"),
        ("Challenge", "Embracing difficult tasks", "Seeking difficult, demanding, and challenging work or situations"),
    ],
    "Universalism": [
        ("Universalism", "Concern for all people and nature", "Care and concern for the welfare of all people and the natural world"),
        ("Justice", "Fairness and equality", "Commitment to fairness, equality, and just treatment for all"),
        ("Equality", "Equal treatment and opportunities", "Belief that all people deserve equal treatment and opportunities"),
        ("Environmental", "Protecting the natural world", "Commitment to environmental protection and sustainability"),
    ],
    "Core": [
        ("Integrity", "Honesty and moral consistency", "Acting in accordance with moral and ethical principles consistently"),
        ("Respect", "Valuing others dignity and worth", "Treating others with dignity, consideration, and appreciation"),
        ("Responsibility", "Accountability for actions and duties", "Taking ownership of actions, decisions, and their consequences"),
        ("Trust", "Reliability and dependability", "Being trustworthy and having confidence in others trustworthiness"),
    ]
}

# Claude AI prompts for values and behavioral coding
CLAUDE_PROMPTS = {
    "open_coding": """You are performing open coding on behavioral event interview transcripts to identify values without constraints.

Analyze the provided text section and identify 3-5 core values that appear to drive the narrative. These can be ANY values - you are not constrained to a predefined list. Look for fundamental motivations, priorities, and driving principles.

Focus on the ROOT values driving the narrative, not surface behaviors or actions.
Consider what fundamental values motivate the decisions, actions, and priorities described.

Return your response as JSON:
{{
  "values": [
    {{"value": "ValueName", "confidence": 0.9, "rationale": "Brief explanation why this value is present", "category": "Suggested category for this value"}},
    {{"value": "ValueName", "confidence": 0.8, "rationale": "Brief explanation why this value is present", "category": "Suggested category for this value"}}
  ]
}}

Text to analyze:
{text_section}""",

    "values_identification": """You are analyzing behavioral event interview transcripts to identify core values being enacted.
For the provided text section, identify 3-5 possible core values from this list:

{values_list}

Focus on the ROOT value driving the narrative, not surface behaviors.
Consider what fundamental values motivate the actions, decisions, and priorities described.

Return your response as JSON:
{{
  "values": [
    {{"value": "ValueName", "confidence": 0.9, "rationale": "Brief explanation why this value is present"}},
    {{"value": "ValueName", "confidence": 0.8, "rationale": "Brief explanation why this value is present"}}
  ]
}}

Text to analyze:
{text_section}""",

    "behavioral_coding": """Using the Behavioral Scale for Enacting Values, suggest an appropriate behavioral score for this sentence.

The selected core value being analyzed is: {selected_value}

Scale:
+3: Extraordinary Enacting (significant personal/professional cost)
+2: Active Enacting (consistent, deliberate)
+1: Compromising (partial, selective)
0: Indifference (apathetic, disengaged)
-1: Capitulating (surrender through inaction)
-2: Active Violation (deliberate contradiction)
-3: Extraordinary Violation (systematic undermining)

Consider how the sentence demonstrates enactment or violation of the value "{selected_value}".

Return your response as JSON:
{{
  "behavioral_score": 2,
  "confidence": 0.85,
  "rationale": "Brief explanation for the behavioral score"
}}

Sentence to analyze:
{sentence_text}""",

    "section_parsing": """Parse this text into logical sections for values coding analysis.
Each section should represent a coherent narrative unit (typically 2-4 sentences) that can be analyzed for a single predominant value.

Return your response as JSON:
{{
  "sections": [
    {{"section_number": 1, "start_pos": 0, "end_pos": 150, "section_text": "First section text...", "section_type": "paragraph"}},
    {{"section_number": 2, "start_pos": 151, "end_pos": 300, "section_text": "Second section text...", "section_type": "paragraph"}}
  ]
}}

Text to parse:
{full_text}""",

    "sentence_parsing": """Parse this section text into individual sentences for behavioral coding analysis.
Each sentence should be a complete grammatical unit that can be analyzed for behavioral enactment.

Return your response as JSON:
{{
  "sentences": [
    {{"sentence_number": 1, "start_pos": 0, "end_pos": 75, "sentence_text": "First sentence."}},
    {{"sentence_number": 2, "start_pos": 76, "end_pos": 150, "sentence_text": "Second sentence."}}
  ]
}}

Section text to parse:
{section_text}""",

    "taxonomy_validation": """You are helping validate a values taxonomy by comparing open-coded values to predefined taxonomy values.

Here are the open-coded values identified for this text:
{open_coded_values}

Here are the predefined taxonomy values:
{taxonomy_values}

For each open-coded value, analyze whether it:
1. Maps directly to a taxonomy value (exact or very close match)
2. Could be categorized under a broader taxonomy value
3. Represents a gap in the taxonomy (new value needed)
4. Is too specific/narrow and should be merged with existing values

Return your response as JSON:
{{
  "mappings": [
    {{
      "open_value": "OpenValueName", 
      "mapping_type": "direct_match|broader_category|taxonomy_gap|too_specific",
      "taxonomy_match": "TaxonomyValueName or null",
      "confidence": 0.9,
      "rationale": "Explanation of the mapping decision"
    }}
  ],
  "taxonomy_recommendations": [
    {{
      "recommendation_type": "add_value|modify_value|merge_values",
      "details": "Specific recommendation for taxonomy improvement"
    }}
  ]
}}

Text context:
{text_section}""",

    "hybrid_coding": """You are performing hybrid values coding - identifying values using both open coding and a predefined taxonomy.

First, identify 3-5 values through open coding (any values you observe):
Then, map those to the closest matches in this taxonomy:

{taxonomy_values}

Return your response as JSON:
{{
  "open_values": [
    {{"value": "OpenValue", "confidence": 0.9, "rationale": "Why this value is present"}}
  ],
  "taxonomy_matches": [
    {{"open_value": "OpenValue", "taxonomy_value": "TaxonomyValue", "match_confidence": 0.8, "mapping_rationale": "Why these match"}}
  ],
  "taxonomy_gaps": [
    {{"missing_value": "ValueNotInTaxonomy", "rationale": "Why this gap exists"}}
  ]
}}

Text to analyze:
{text_section}"""
}

# Default export templates
DEFAULT_EXPORT_TEMPLATES = {
    "values_summary": {
        "name": "Values Summary",
        "type": "values_summary",
        "format": "csv",
        "config": {
            "include_fields": ["section_number", "section_text", "selected_value", "confidence_level", "coder_notes"],
            "group_by": "selected_value",
            "include_statistics": True
        }
    },
    "behavioral_summary": {
        "name": "Behavioral Summary",
        "type": "behavioral_summary", 
        "format": "csv",
        "config": {
            "include_fields": ["sentence_number", "sentence_text", "behavioral_score", "selected_value", "coder_rationale"],
            "group_by": "behavioral_score",
            "include_statistics": True
        }
    },
    "combined_analysis": {
        "name": "Combined Values & Behavioral Analysis",
        "type": "combined",
        "format": "excel",
        "config": {
            "include_values_sheet": True,
            "include_behavioral_sheet": True,
            "include_summary_sheet": True,
            "include_visualizations": True
        }
    },
    "detailed_report": {
        "name": "Detailed Coding Report",
        "type": "detailed",
        "format": "pdf",
        "config": {
            "include_methodology": True,
            "include_inter_rater_reliability": True,
            "include_code_frequency_tables": True,
            "include_narrative_examples": True
        }
    }
}

# UI Color scheme for values categories
VALUES_CATEGORY_COLORS = {
    "Achievement": "#FF6B6B",      # Red
    "Benevolence": "#4ECDC4",      # Teal  
    "Conformity": "#45B7D1",       # Blue
    "Hedonism": "#FFA07A",         # Orange
    "Power": "#9B59B6",            # Purple
    "Security": "#F39C12",         # Gold
    "Self-Direction": "#2ECC71",   # Green
    "Stimulation": "#E74C3C",      # Dark Red
    "Universalism": "#3498DB",     # Dark Blue
    "Core": "#34495E"              # Dark Gray
}

# Default confidence thresholds for auto-suggestions
CONFIDENCE_THRESHOLDS = {
    "values_auto_accept": 0.95,     # Auto-accept Claude values suggestions above this threshold
    "behavioral_auto_accept": 0.95,  # Auto-accept Claude behavioral suggestions above this threshold
    "values_show_suggestion": 0.60,  # Show values suggestions above this threshold
    "behavioral_show_suggestion": 0.60  # Show behavioral suggestions above this threshold
}