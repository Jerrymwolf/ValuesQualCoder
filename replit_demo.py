"""
Replit Demo for Values and Behavioral Enactment Coder
Simplified demo without full QualCoder dependencies
"""

import json
import asyncio
import os
from datetime import datetime
from typing import List, Dict, Any

# Simplified models for demo
class SimpleOpenValue:
    def __init__(self, value_name, category, confidence, rationale):
        self.value_name = value_name
        self.category = category
        self.confidence = confidence
        self.rationale = rationale

class SimpleTaxonomyValue:
    def __init__(self, value_name, category, description):
        self.value_name = value_name
        self.category = category
        self.description = description

# Sample taxonomy (subset of the full 32 values)
SAMPLE_TAXONOMY = [
    SimpleTaxonomyValue("Integrity", "Core", "Acting in accordance with moral and ethical principles consistently"),
    SimpleTaxonomyValue("Excellence", "Achievement", "Pursuing the highest quality and standards"),
    SimpleTaxonomyValue("Service", "Benevolence", "Helping and supporting others"),
    SimpleTaxonomyValue("Responsibility", "Core", "Taking ownership of actions, decisions, and their consequences"),
    SimpleTaxonomyValue("Competence", "Achievement", "Developing and demonstrating skills"),
    SimpleTaxonomyValue("Collaboration", "Benevolence", "Working effectively with others toward common goals"),
]

# AI Prompts (simplified for demo)
DEMO_PROMPTS = {
    "open_coding": """You are performing open coding on behavioral event interview transcripts to identify values without constraints.

Analyze the provided text section and identify 3-5 core values that appear to drive the narrative. These can be ANY values - you are not constrained to a predefined list.

Return your response as JSON:
{
  "values": [
    {"value": "ValueName", "confidence": 0.9, "rationale": "Brief explanation", "category": "Suggested category"}
  ]
}

Text to analyze:
{text_section}""",

    "taxonomy_coding": """Analyze this text for values from this predefined list:

{taxonomy_list}

Return your response as JSON:
{
  "values": [
    {"value": "ValueName", "confidence": 0.9, "rationale": "Brief explanation"}
  ]
}

Text to analyze:
{text_section}"""
}

class ValuesAnalyzer:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        
    def analyze_open_coding(self, text):
        """Simulate open coding analysis"""
        print(f"Analyzing text (Open Coding): {text[:100]}...")
        
        # For demo - simulate AI response
        # In real version, this would call Claude API
        simulated_results = [
            SimpleOpenValue("Integrity", "Core", 0.95, "Mentions doing right thing when no one watching"),
            SimpleOpenValue("Excellence", "Achievement", 0.87, "Emphasis on high standards and quality work"),
            SimpleOpenValue("Mentorship", "Benevolence", 0.82, "Helping colleague with complex project"),
            SimpleOpenValue("Accountability", "Core", 0.78, "Taking personal responsibility for investigation")
        ]
        
        return simulated_results
    
    def analyze_taxonomy_coding(self, text):
        """Simulate taxonomy-based coding"""
        print(f"Analyzing text (Taxonomy): {text[:100]}...")
        
        # Simulate matching to predefined taxonomy
        matches = []
        if "integrity" in text.lower():
            matches.append({"value": "Integrity", "confidence": 0.95, "rationale": "Direct mention of integrity"})
        if "excellence" in text.lower() or "quality" in text.lower():
            matches.append({"value": "Excellence", "confidence": 0.87, "rationale": "Focus on quality and standards"})
        if "help" in text.lower() or "mentor" in text.lower():
            matches.append({"value": "Service", "confidence": 0.82, "rationale": "Helping and mentoring others"})
        if "responsibility" in text.lower():
            matches.append({"value": "Responsibility", "confidence": 0.78, "rationale": "Taking personal responsibility"})
            
        return matches
    
    def compare_approaches(self, text):
        """Compare open coding vs taxonomy approaches"""
        print("\n" + "="*60)
        print("COMPARATIVE ANALYSIS: Open Coding vs Taxonomy")
        print("="*60)
        
        # Open coding
        print("\n1. OPEN CODING RESULTS:")
        print("-" * 30)
        open_results = self.analyze_open_coding(text)
        for result in open_results:
            print(f"• {result.value_name} ({result.category}) - {result.confidence:.1%}")
            print(f"  → {result.rationale}")
        
        # Taxonomy coding
        print("\n2. TAXONOMY CODING RESULTS:")
        print("-" * 30)
        taxonomy_results = self.analyze_taxonomy_coding(text)
        for result in taxonomy_results:
            print(f"• {result['value']} - {result['confidence']:.1%}")
            print(f"  → {result['rationale']}")
        
        # Comparison
        print("\n3. COMPARISON ANALYSIS:")
        print("-" * 30)
        
        open_values = set(r.value_name for r in open_results)
        taxonomy_values = set(r['value'] for r in taxonomy_results)
        
        matches = open_values.intersection(taxonomy_values)
        open_only = open_values - taxonomy_values
        taxonomy_only = taxonomy_values - open_values
        
        print(f"✓ Values found in BOTH approaches: {', '.join(matches) if matches else 'None'}")
        print(f"• Values found ONLY in open coding: {', '.join(open_only) if open_only else 'None'}")
        print(f"• Values found ONLY in taxonomy: {', '.join(taxonomy_only) if taxonomy_only else 'None'}")
        
        # Gap analysis
        print(f"\n📊 TAXONOMY VALIDATION INSIGHTS:")
        print(f"   - Taxonomy coverage: {len(matches)}/{len(open_values)} ({len(matches)/len(open_values)*100:.1f}%)")
        print(f"   - New values identified: {len(open_only)}")
        print(f"   - Unused taxonomy values: {len(SAMPLE_TAXONOMY) - len(taxonomy_values)}")
        
        if open_only:
            print(f"\n💡 RECOMMENDATIONS:")
            print(f"   Consider adding these values to taxonomy: {', '.join(open_only)}")

def demo_basic_analysis():
    """Run a basic analysis demo"""
    
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
    
    analyzer = ValuesAnalyzer()
    analyzer.compare_approaches(sample_text)

def demo_behavioral_scale():
    """Demonstrate the behavioral scale"""
    print("\n" + "="*60)
    print("BEHAVIORAL ENACTMENT SCALE DEMO")
    print("="*60)
    
    scale = {
        -3: {"name": "Extraordinary Violation", "description": "Systematically undermining values"},
        -2: {"name": "Active Violation", "description": "Deliberately contradicting values"},  
        -1: {"name": "Capitulating", "description": "Surrendering through inaction"},
        0: {"name": "Indifference", "description": "Showing no concern for values"},
        1: {"name": "Compromising", "description": "Partial, selective enactment"},
        2: {"name": "Active Enacting", "description": "Consistent, deliberate alignment"},
        3: {"name": "Extraordinary Enacting", "description": "Going above and beyond, with personal cost"}
    }
    
    print("\nBEHAVIORAL SCALE FOR ENACTING VALUES:")
    print("-" * 40)
    
    for score, info in scale.items():
        symbol = "🟢" if score > 0 else "🔴" if score < 0 else "⚪"
        print(f"{symbol} {score:+2d}: {info['name']}")
        print(f"     → {info['description']}")
    
    # Example sentence analysis
    sentences = [
        ("I took personal responsibility to investigate the issue", "Integrity", 2),
        ("I always push myself to deliver highest quality work", "Excellence", 2), 
        ("I volunteered my time to mentor a struggling colleague", "Service", 3),
        ("I ignored the problem hoping someone else would handle it", "Responsibility", -1)
    ]
    
    print(f"\n📝 EXAMPLE SENTENCE CODING:")
    print("-" * 40)
    
    for sentence, value, score in sentences:
        scale_info = scale[score]
        symbol = "🟢" if score > 0 else "🔴" if score < 0 else "⚪"
        print(f"\n'{sentence}'")
        print(f"Value: {value} | Score: {score:+2d} ({scale_info['name']}) {symbol}")

def main_menu():
    """Interactive demo menu"""
    while True:
        print("\n" + "="*60)
        print("VALUES AND BEHAVIORAL ENACTMENT CODER - DEMO")
        print("="*60)
        print("\nChoose a demo:")
        print("1. Basic Values Analysis (Open vs Taxonomy)")
        print("2. Behavioral Scale Demonstration") 
        print("3. Custom Text Analysis")
        print("4. Show Sample Taxonomy")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            demo_basic_analysis()
        elif choice == '2':
            demo_behavioral_scale()
        elif choice == '3':
            custom_text = input("\nEnter your text to analyze: ")
            if custom_text.strip():
                analyzer = ValuesAnalyzer()
                analyzer.compare_approaches(custom_text)
        elif choice == '4':
            print("\n" + "="*60)
            print("SAMPLE TAXONOMY (6 of 32 values)")
            print("="*60)
            for val in SAMPLE_TAXONOMY:
                print(f"• {val.value_name} ({val.category})")
                print(f"  → {val.description}")
        elif choice == '5':
            print("\nGoodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    print("Values and Behavioral Enactment Coder - Replit Demo")
    print("This demonstrates the core concepts without full QualCoder integration")
    
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("\n⚠️  Note: No Anthropic API key found. Demo will use simulated results.")
        print("To use real AI: Set ANTHROPIC_API_KEY environment variable")
    
    main_menu()