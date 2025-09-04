"""
Analysis MCP Server
Provides statistical analysis and reporting for values and behavioral coding
"""

import json
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from collections import Counter, defaultdict
import io
import base64
from pathlib import Path

try:
    from mcp.server import Server
    from mcp.types import Tool, TextContent
    import mcp.server.stdio
except ImportError:
    print("MCP not available - install with: pip install mcp")
    Server = None
    Tool = None
    TextContent = None

class AnalysisServer:
    def __init__(self, database_path: str = None):
        self.database_path = database_path or "values_behavioral.db"
        self.server = Server("analysis") if Server else None
        
        # Set up matplotlib for non-interactive use
        plt.ioff()
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        
        if self.server:
            self._register_tools()
    
    def _register_tools(self):
        """Register MCP tools"""
        
        @self.server.call_tool()
        async def calculate_frequencies(session_id: Optional[int] = None, value_category: Optional[str] = None) -> List[TextContent]:
            """Calculate frequency statistics for values coding"""
            conn = sqlite3.connect(self.database_path)
            try:
                cursor = conn.cursor()
                
                # Base query for values coding
                base_query = """
                    SELECT 
                        COALESCE(ctv.value_name, vc.custom_value_name) as value_name,
                        COALESCE(ctv.value_category, 'Custom') as value_category,
                        COUNT(*) as frequency,
                        AVG(CASE WHEN csvs.confidence_score IS NOT NULL 
                            THEN csvs.confidence_score ELSE 0.8 END) as avg_confidence
                    FROM values_coding vc
                    LEFT JOIN core_values_taxonomy ctv ON vc.value_id = ctv.value_id
                    LEFT JOIN document_sections ds ON vc.section_id = ds.section_id
                    LEFT JOIN claude_values_suggestions csvs ON ds.section_id = csvs.section_id 
                        AND csvs.value_id = vc.value_id
                    WHERE 1=1
                """
                
                params = []
                if session_id:
                    base_query += " AND ds.session_id = ?"
                    params.append(session_id)
                
                if value_category:
                    base_query += " AND (ctv.value_category = ? OR vc.custom_value_name IS NOT NULL)"
                    params.append(value_category)
                
                base_query += " GROUP BY value_name, value_category ORDER BY frequency DESC"
                
                cursor.execute(base_query, params)
                value_frequencies = []
                
                for row in cursor.fetchall():
                    value_frequencies.append({
                        'value_name': row[0],
                        'value_category': row[1],
                        'frequency': row[2],
                        'average_confidence': round(row[3], 3)
                    })
                
                # Category frequency analysis
                category_freq = defaultdict(int)
                category_confidence = defaultdict(list)
                
                for item in value_frequencies:
                    category_freq[item['value_category']] += item['frequency']
                    category_confidence[item['value_category']].append(item['average_confidence'])
                
                category_stats = []
                for category, freq in category_freq.items():
                    avg_confidence = np.mean(category_confidence[category]) if category_confidence[category] else 0
                    category_stats.append({
                        'category': category,
                        'total_frequency': freq,
                        'unique_values': len([v for v in value_frequencies if v['value_category'] == category]),
                        'average_confidence': round(avg_confidence, 3)
                    })
                
                category_stats.sort(key=lambda x: x['total_frequency'], reverse=True)
                
                result = {
                    'value_frequencies': value_frequencies,
                    'category_frequencies': category_stats,
                    'total_coded_sections': sum(v['frequency'] for v in value_frequencies),
                    'unique_values_count': len(value_frequencies),
                    'unique_categories_count': len(category_stats),
                    'analysis_timestamp': datetime.now().isoformat()
                }
                
                return [TextContent(
                    type="text",
                    text=json.dumps(result, indent=2)
                )]
            finally:
                conn.close()
        
        @self.server.call_tool()
        async def perform_gap_analysis(open_coded_values: List[str], taxonomy_values: List[str]) -> List[TextContent]:
            """Analyze gaps between open coding and taxonomy"""
            
            open_set = set(v.lower().strip() for v in open_coded_values)
            taxonomy_set = set(v.lower().strip() for v in taxonomy_values)
            
            # Find overlaps and gaps
            exact_matches = open_set.intersection(taxonomy_set)
            open_only = open_set - taxonomy_set
            taxonomy_unused = taxonomy_set - open_set
            
            # Fuzzy matching for similar values
            from difflib import SequenceMatcher
            
            def fuzzy_match(value, target_set, threshold=0.7):
                best_match = None
                best_ratio = 0
                for target in target_set:
                    ratio = SequenceMatcher(None, value, target).ratio()
                    if ratio > best_ratio and ratio >= threshold:
                        best_match = target
                        best_ratio = ratio
                return best_match, best_ratio
            
            fuzzy_matches = []
            potential_matches = []
            
            for open_value in open_only:
                match, ratio = fuzzy_match(open_value, taxonomy_unused)
                if match:
                    if ratio >= 0.8:
                        fuzzy_matches.append({
                            'open_value': open_value,
                            'taxonomy_match': match,
                            'similarity': round(ratio, 3)
                        })
                    else:
                        potential_matches.append({
                            'open_value': open_value,
                            'taxonomy_match': match,
                            'similarity': round(ratio, 3)
                        })
            
            # Calculate coverage metrics
            total_open = len(open_set)
            covered_exact = len(exact_matches)
            covered_fuzzy = len(fuzzy_matches)
            total_covered = covered_exact + covered_fuzzy
            
            coverage_rate = total_covered / total_open if total_open > 0 else 0
            taxonomy_usage = (len(taxonomy_set) - len(taxonomy_unused)) / len(taxonomy_set) if len(taxonomy_set) > 0 else 0
            
            # Identify new value suggestions
            truly_new_values = open_only - set(fm['open_value'] for fm in fuzzy_matches + potential_matches)
            
            result = {
                'gap_analysis_summary': {
                    'open_coded_values_count': len(open_set),
                    'taxonomy_values_count': len(taxonomy_set),
                    'exact_matches': len(exact_matches),
                    'fuzzy_matches': len(fuzzy_matches),
                    'coverage_rate': round(coverage_rate, 3),
                    'taxonomy_usage_rate': round(taxonomy_usage, 3)
                },
                'exact_matches': list(exact_matches),
                'fuzzy_matches': fuzzy_matches,
                'potential_matches': potential_matches,
                'open_only_values': list(open_only),
                'unused_taxonomy_values': list(taxonomy_unused),
                'new_value_candidates': list(truly_new_values),
                'recommendations': self._generate_gap_recommendations(
                    coverage_rate, len(truly_new_values), len(taxonomy_unused)
                )
            }
            
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
        
        @self.server.call_tool()
        async def generate_statistics(session_id: Optional[int] = None) -> List[TextContent]:
            """Generate comprehensive statistical report"""
            conn = sqlite3.connect(self.database_path)
            try:
                cursor = conn.cursor()
                
                # Values coding statistics
                values_query = """
                    SELECT 
                        COUNT(*) as total_sections,
                        COUNT(DISTINCT vc.value_id) + COUNT(DISTINCT vc.custom_value_name) as unique_values,
                        AVG(CASE WHEN csvs.confidence_score IS NOT NULL 
                            THEN csvs.confidence_score ELSE 0.8 END) as avg_confidence,
                        COUNT(CASE WHEN vc.confidence_level = 'high' THEN 1 END) as high_confidence,
                        COUNT(CASE WHEN vc.confidence_level = 'medium' THEN 1 END) as medium_confidence,
                        COUNT(CASE WHEN vc.confidence_level = 'low' THEN 1 END) as low_confidence
                    FROM values_coding vc
                    LEFT JOIN document_sections ds ON vc.section_id = ds.section_id
                    LEFT JOIN claude_values_suggestions csvs ON ds.section_id = csvs.section_id
                """
                
                if session_id:
                    values_query += " WHERE ds.session_id = ?"
                    cursor.execute(values_query, (session_id,))
                else:
                    cursor.execute(values_query)
                
                values_stats = cursor.fetchone()
                
                # Behavioral coding statistics
                behavioral_query = """
                    SELECT 
                        COUNT(*) as total_sentences,
                        AVG(behavioral_score) as avg_score,
                        MIN(behavioral_score) as min_score,
                        MAX(behavioral_score) as max_score,
                        COUNT(CASE WHEN behavioral_score > 0 THEN 1 END) as positive_scores,
                        COUNT(CASE WHEN behavioral_score < 0 THEN 1 END) as negative_scores,
                        COUNT(CASE WHEN behavioral_score = 0 THEN 1 END) as neutral_scores
                    FROM behavioral_coding bc
                    LEFT JOIN document_sentences ds ON bc.sentence_id = ds.sentence_id
                    LEFT JOIN document_sections sec ON ds.section_id = sec.section_id
                """
                
                if session_id:
                    behavioral_query += " WHERE sec.session_id = ?"
                    cursor.execute(behavioral_query, (session_id,))
                else:
                    cursor.execute(behavioral_query)
                
                behavioral_stats = cursor.fetchone()
                
                # Behavioral score distribution
                score_dist_query = """
                    SELECT behavioral_score, COUNT(*) as frequency
                    FROM behavioral_coding bc
                    LEFT JOIN document_sentences ds ON bc.sentence_id = ds.sentence_id
                    LEFT JOIN document_sections sec ON ds.section_id = sec.section_id
                """
                
                if session_id:
                    score_dist_query += " WHERE sec.session_id = ? GROUP BY behavioral_score ORDER BY behavioral_score"
                    cursor.execute(score_dist_query, (session_id,))
                else:
                    score_dist_query += " GROUP BY behavioral_score ORDER BY behavioral_score"
                    cursor.execute(score_dist_query)
                
                score_distribution = dict(cursor.fetchall())
                
                # Progress statistics
                progress_query = """
                    SELECT 
                        COUNT(*) as total_sessions,
                        SUM(total_sections) as total_sections,
                        SUM(sections_values_coded) as completed_values,
                        SUM(total_sentences) as total_sentences,
                        SUM(sentences_behavioral_coded) as completed_behavioral
                    FROM coding_progress
                """
                
                if session_id:
                    progress_query += " WHERE session_id = ?"
                    cursor.execute(progress_query, (session_id,))
                else:
                    cursor.execute(progress_query)
                
                progress_stats = cursor.fetchone()
                
                result = {
                    'values_coding': {
                        'total_coded_sections': values_stats[0] or 0,
                        'unique_values_identified': values_stats[1] or 0,
                        'average_ai_confidence': round(values_stats[2] or 0, 3),
                        'confidence_distribution': {
                            'high': values_stats[3] or 0,
                            'medium': values_stats[4] or 0,
                            'low': values_stats[5] or 0
                        }
                    },
                    'behavioral_coding': {
                        'total_coded_sentences': behavioral_stats[0] or 0,
                        'average_behavioral_score': round(behavioral_stats[1] or 0, 2),
                        'score_range': {
                            'min': behavioral_stats[2] or 0,
                            'max': behavioral_stats[3] or 0
                        },
                        'score_distribution': {
                            'positive': behavioral_stats[4] or 0,
                            'negative': behavioral_stats[5] or 0,
                            'neutral': behavioral_stats[6] or 0
                        },
                        'detailed_distribution': score_distribution
                    },
                    'progress_overview': {
                        'total_sessions': progress_stats[0] or 0,
                        'total_sections': progress_stats[1] or 0,
                        'values_completion_rate': round((progress_stats[2] or 0) / (progress_stats[1] or 1), 3),
                        'behavioral_completion_rate': round((progress_stats[4] or 0) / (progress_stats[3] or 1), 3)
                    },
                    'analysis_timestamp': datetime.now().isoformat()
                }
                
                return [TextContent(
                    type="text",
                    text=json.dumps(result, indent=2)
                )]
            finally:
                conn.close()
        
        @self.server.call_tool()
        async def create_visualizations(data_type: str, session_id: Optional[int] = None) -> List[TextContent]:
            """Create visualization charts for analysis data"""
            
            if data_type == "values_frequency":
                chart_data = await self._create_values_frequency_chart(session_id)
            elif data_type == "behavioral_distribution":
                chart_data = await self._create_behavioral_distribution_chart(session_id)
            elif data_type == "category_comparison":
                chart_data = await self._create_category_comparison_chart(session_id)
            elif data_type == "progress_overview":
                chart_data = await self._create_progress_overview_chart(session_id)
            else:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": f"Unknown visualization type: {data_type}"})
                )]
            
            return [TextContent(
                type="text",
                text=json.dumps(chart_data, indent=2)
            )]
        
        @self.server.call_tool()
        async def export_results(format_type: str, session_id: Optional[int] = None, include_raw_data: bool = False) -> List[TextContent]:
            """Export analysis results in various formats"""
            
            # Get all analysis data
            frequencies = await calculate_frequencies(session_id)
            statistics = await generate_statistics(session_id)
            
            freq_data = json.loads(frequencies[0].text)
            stats_data = json.loads(statistics[0].text)
            
            if format_type == "json":
                export_data = {
                    'export_metadata': {
                        'format': 'json',
                        'session_id': session_id,
                        'export_timestamp': datetime.now().isoformat(),
                        'includes_raw_data': include_raw_data
                    },
                    'frequency_analysis': freq_data,
                    'statistical_summary': stats_data
                }
                
                if include_raw_data:
                    export_data['raw_data'] = await self._get_raw_data(session_id)
                
                return [TextContent(
                    type="text", 
                    text=json.dumps(export_data, indent=2)
                )]
            
            elif format_type == "csv":
                csv_data = await self._generate_csv_export(freq_data, stats_data, session_id)
                return [TextContent(type="text", text=csv_data)]
            
            elif format_type == "summary_report":
                report = await self._generate_summary_report(freq_data, stats_data, session_id)
                return [TextContent(type="text", text=report)]
            
            else:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": f"Unsupported export format: {format_type}"})
                )]
    
    async def _create_values_frequency_chart(self, session_id: Optional[int] = None) -> Dict[str, Any]:
        """Create values frequency visualization"""
        conn = sqlite3.connect(self.database_path)
        try:
            # Get frequency data
            cursor = conn.cursor()
            query = """
                SELECT 
                    COALESCE(ctv.value_name, vc.custom_value_name) as value_name,
                    COUNT(*) as frequency
                FROM values_coding vc
                LEFT JOIN core_values_taxonomy ctv ON vc.value_id = ctv.value_id
                LEFT JOIN document_sections ds ON vc.section_id = ds.section_id
                WHERE 1=1
            """
            
            params = []
            if session_id:
                query += " AND ds.session_id = ?"
                params.append(session_id)
            
            query += " GROUP BY value_name ORDER BY frequency DESC LIMIT 20"
            cursor.execute(query, params)
            
            data = cursor.fetchall()
            if not data:
                return {"error": "No data available for visualization"}
            
            values, frequencies = zip(*data)
            
            # Create bar chart
            plt.figure(figsize=(12, 8))
            plt.bar(range(len(values)), frequencies)
            plt.xticks(range(len(values)), values, rotation=45, ha='right')
            plt.ylabel('Frequency')
            plt.title('Most Frequently Identified Values')
            plt.tight_layout()
            
            # Save to base64
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.read()).decode()
            plt.close()
            
            return {
                'chart_type': 'values_frequency_bar',
                'chart_data': img_base64,
                'data_points': len(data),
                'top_value': values[0],
                'top_frequency': frequencies[0]
            }
        finally:
            conn.close()
    
    async def _create_behavioral_distribution_chart(self, session_id: Optional[int] = None) -> Dict[str, Any]:
        """Create behavioral score distribution visualization"""
        conn = sqlite3.connect(self.database_path)
        try:
            cursor = conn.cursor()
            query = """
                SELECT behavioral_score, COUNT(*) as frequency
                FROM behavioral_coding bc
                LEFT JOIN document_sentences ds ON bc.sentence_id = ds.sentence_id
                LEFT JOIN document_sections sec ON ds.section_id = sec.section_id
                WHERE 1=1
            """
            
            params = []
            if session_id:
                query += " AND sec.session_id = ?"
                params.append(session_id)
            
            query += " GROUP BY behavioral_score ORDER BY behavioral_score"
            cursor.execute(query, params)
            
            data = dict(cursor.fetchall())
            if not data:
                return {"error": "No behavioral data available"}
            
            # Ensure all scores -3 to 3 are represented
            scores = range(-3, 4)
            frequencies = [data.get(score, 0) for score in scores]
            
            # Create histogram
            plt.figure(figsize=(10, 6))
            colors = ['red' if s < 0 else 'gray' if s == 0 else 'green' for s in scores]
            plt.bar(scores, frequencies, color=colors, alpha=0.7)
            plt.xlabel('Behavioral Score')
            plt.ylabel('Frequency')
            plt.title('Distribution of Behavioral Enactment Scores')
            plt.xticks(scores)
            plt.grid(axis='y', alpha=0.3)
            
            # Save to base64
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.read()).decode()
            plt.close()
            
            return {
                'chart_type': 'behavioral_distribution_histogram',
                'chart_data': img_base64,
                'total_sentences': sum(frequencies),
                'positive_percentage': round(sum(frequencies[4:]) / sum(frequencies) * 100, 1) if sum(frequencies) > 0 else 0,
                'negative_percentage': round(sum(frequencies[:3]) / sum(frequencies) * 100, 1) if sum(frequencies) > 0 else 0
            }
        finally:
            conn.close()
    
    async def _create_category_comparison_chart(self, session_id: Optional[int] = None) -> Dict[str, Any]:
        """Create category comparison visualization"""
        conn = sqlite3.connect(self.database_path)
        try:
            cursor = conn.cursor()
            query = """
                SELECT 
                    COALESCE(ctv.value_category, 'Custom') as category,
                    COUNT(*) as frequency
                FROM values_coding vc
                LEFT JOIN core_values_taxonomy ctv ON vc.value_id = ctv.value_id
                LEFT JOIN document_sections ds ON vc.section_id = ds.section_id
                WHERE 1=1
            """
            
            params = []
            if session_id:
                query += " AND ds.session_id = ?"
                params.append(session_id)
            
            query += " GROUP BY category ORDER BY frequency DESC"
            cursor.execute(query, params)
            
            data = cursor.fetchall()
            if not data:
                return {"error": "No category data available"}
            
            categories, frequencies = zip(*data)
            
            # Create pie chart
            plt.figure(figsize=(10, 8))
            plt.pie(frequencies, labels=categories, autopct='%1.1f%%', startangle=90)
            plt.title('Values Distribution by Category')
            plt.axis('equal')
            
            # Save to base64
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.read()).decode()
            plt.close()
            
            return {
                'chart_type': 'category_pie_chart',
                'chart_data': img_base64,
                'categories_count': len(categories),
                'dominant_category': categories[0],
                'dominant_percentage': round(frequencies[0] / sum(frequencies) * 100, 1)
            }
        finally:
            conn.close()
    
    async def _create_progress_overview_chart(self, session_id: Optional[int] = None) -> Dict[str, Any]:
        """Create progress overview visualization"""
        conn = sqlite3.connect(self.database_path)
        try:
            cursor = conn.cursor()
            query = """
                SELECT 
                    total_sections,
                    sections_values_coded,
                    total_sentences,
                    sentences_behavioral_coded
                FROM coding_progress
                WHERE 1=1
            """
            
            params = []
            if session_id:
                query += " AND session_id = ?"
                params.append(session_id)
            
            cursor.execute(query, params)
            data = cursor.fetchall()
            
            if not data:
                return {"error": "No progress data available"}
            
            # Aggregate data
            total_sections = sum(row[0] for row in data)
            completed_values = sum(row[1] for row in data)
            total_sentences = sum(row[2] for row in data)
            completed_behavioral = sum(row[3] for row in data)
            
            # Create progress bars
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
            
            # Values coding progress
            values_progress = completed_values / total_sections if total_sections > 0 else 0
            ax1.barh(['Values Coding'], [values_progress], color='blue', alpha=0.7)
            ax1.set_xlim(0, 1)
            ax1.set_title('Values Coding Progress')
            ax1.set_xlabel('Completion Rate')
            
            # Behavioral coding progress
            behavioral_progress = completed_behavioral / total_sentences if total_sentences > 0 else 0
            ax2.barh(['Behavioral Coding'], [behavioral_progress], color='orange', alpha=0.7)
            ax2.set_xlim(0, 1)
            ax2.set_title('Behavioral Coding Progress')
            ax2.set_xlabel('Completion Rate')
            
            plt.tight_layout()
            
            # Save to base64
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.read()).decode()
            plt.close()
            
            return {
                'chart_type': 'progress_bars',
                'chart_data': img_base64,
                'values_completion': round(values_progress * 100, 1),
                'behavioral_completion': round(behavioral_progress * 100, 1),
                'total_sections': total_sections,
                'total_sentences': total_sentences
            }
        finally:
            conn.close()
    
    def _generate_gap_recommendations(self, coverage_rate: float, new_values_count: int, unused_count: int) -> List[str]:
        """Generate recommendations based on gap analysis"""
        recommendations = []
        
        if coverage_rate < 0.7:
            recommendations.append(f"Low taxonomy coverage ({coverage_rate:.1%}). Consider expanding or refining the taxonomy.")
        
        if new_values_count > 5:
            recommendations.append(f"Many new values identified ({new_values_count}). Review for potential taxonomy additions.")
        
        if unused_count > 10:
            recommendations.append(f"Many unused taxonomy values ({unused_count}). Consider taxonomy pruning or additional coding.")
        
        if coverage_rate > 0.9:
            recommendations.append("Excellent taxonomy coverage. Current taxonomy appears comprehensive.")
        
        return recommendations
    
    async def _get_raw_data(self, session_id: Optional[int] = None) -> Dict[str, Any]:
        """Get raw coding data for export"""
        conn = sqlite3.connect(self.database_path)
        try:
            # This would implement full raw data extraction
            # Simplified for demo
            return {"note": "Raw data export not implemented in demo version"}
        finally:
            conn.close()
    
    async def _generate_csv_export(self, freq_data: Dict, stats_data: Dict, session_id: Optional[int]) -> str:
        """Generate CSV format export"""
        csv_lines = ["# Values and Behavioral Analysis Export"]
        csv_lines.append("# Generated: " + datetime.now().isoformat())
        csv_lines.append("")
        
        # Values frequency data
        csv_lines.append("## Values Frequency")
        csv_lines.append("Value Name,Category,Frequency,Average Confidence")
        
        for item in freq_data.get('value_frequencies', []):
            csv_lines.append(f"{item['value_name']},{item['value_category']},{item['frequency']},{item['average_confidence']}")
        
        return "\n".join(csv_lines)
    
    async def _generate_summary_report(self, freq_data: Dict, stats_data: Dict, session_id: Optional[int]) -> str:
        """Generate summary report"""
        report_lines = ["# Values and Behavioral Enactment Analysis Report"]
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Summary statistics
        values_stats = stats_data.get('values_coding', {})
        behavioral_stats = stats_data.get('behavioral_coding', {})
        
        report_lines.extend([
            "## Executive Summary",
            f"- Total coded sections: {values_stats.get('total_coded_sections', 0)}",
            f"- Unique values identified: {values_stats.get('unique_values_identified', 0)}",
            f"- Total coded sentences: {behavioral_stats.get('total_coded_sentences', 0)}",
            f"- Average behavioral score: {behavioral_stats.get('average_behavioral_score', 0)}",
            ""
        ])
        
        return "\n".join(report_lines)

async def main():
    """Run the Analysis MCP Server"""
    if not Server:
        print("MCP library not available. Install with: pip install mcp")
        return
    
    server = AnalysisServer()
    
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.server.run(
            read_stream,
            write_stream,
            server.server.create_initialization_options()
        )

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())