"""
Values and Behavioral Enactment Coder - Open Coding Service
Service layer for open coding and taxonomy validation functionality
"""

import sqlite3
import json
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple

from .values_behavioral_service import ValuesBehavioralService
from .values_behavioral_constants import CodingMode, CLAUDE_PROMPTS, CORE_VALUES_TAXONOMY
from .values_behavioral_open_coding import (
    OpenCodedValue, TaxonomyMapping, TaxonomyRecommendation, HybridCodingResult,
    OpenCodingStatistics, TaxonomyValidationSession, MappingType, RecommendationType,
    OpenCodingException, TaxonomyValidationException
)
from .values_behavioral_models import ValuesCodingSession, DocumentSection, DatabaseError

logger = logging.getLogger(__name__)


class OpenCodingService(ValuesBehavioralService):
    """Extended service class supporting open coding and taxonomy validation"""
    
    def __init__(self, app):
        """Initialize with app instance"""
        super().__init__(app)
    
    async def get_open_coding_suggestions(self, section_id: int, section_text: str, 
                                        use_cache: bool = True) -> List[OpenCodedValue]:
        """Get open coding suggestions from Claude (unrestricted values)"""
        try:
            # Check if we have AI service
            if not hasattr(self.app, 'ai') or not self.app.ai.is_ready():
                raise OpenCodingException("AI service not available")
            
            # Create prompt for open coding
            prompt = CLAUDE_PROMPTS["open_coding"].format(text_section=section_text)
            
            # Get AI response
            response = await self._get_ai_completion(prompt)
            
            # Parse JSON response
            try:
                result = json.loads(response)
                suggestions = result.get('values', [])
            except json.JSONDecodeError:
                import json_repair
                repaired = json_repair.repair_json(response)
                result = json.loads(repaired)
                suggestions = result.get('values', [])
            
            # Process suggestions into OpenCodedValue objects
            open_values = []
            for suggestion in suggestions:
                open_value = OpenCodedValue(
                    section_id=section_id,
                    value_name=suggestion.get('value', ''),
                    suggested_category=suggestion.get('category', ''),
                    confidence_score=float(suggestion.get('confidence', 0.0)),
                    rationale=suggestion.get('rationale', ''),
                    coded_date=datetime.now(),
                    coder_name=self.app.settings.get('codername', 'AI'),
                    model_version=getattr(self.app.ai, 'model_version', 'claude-3-5-sonnet'),
                    is_validated=False
                )
                open_values.append(open_value)
            
            logger.info(f"Generated {len(open_values)} open coding suggestions for section {section_id}")
            return open_values
            
        except Exception as e:
            logger.error(f"Error getting open coding suggestions: {e}")
            raise OpenCodingException(f"Failed to get open coding suggestions: {e}")
    
    def save_open_coded_value(self, open_value: OpenCodedValue) -> OpenCodedValue:
        """Save an open-coded value to the database"""
        try:
            cursor = self.conn.cursor()
            
            cursor.execute("""
                INSERT INTO open_coded_values 
                (section_id, value_name, suggested_category, confidence_score, rationale, 
                 coded_date, coder_name, model_version, is_validated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                open_value.section_id, open_value.value_name, open_value.suggested_category,
                open_value.confidence_score, open_value.rationale, 
                open_value.coded_date.isoformat() if open_value.coded_date else None,
                open_value.coder_name, open_value.model_version, open_value.is_validated
            ))
            
            open_value.open_value_id = cursor.lastrowid
            self.conn.commit()
            
            logger.info(f"Saved open coded value: {open_value.value_name}")
            return open_value
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error saving open coded value: {e}")
            raise DatabaseError(f"Failed to save open coded value: {e}")
    
    def get_open_coded_values(self, section_id: Optional[int] = None, 
                            coder_name: Optional[str] = None) -> List[OpenCodedValue]:
        """Retrieve open-coded values"""
        try:
            cursor = self.conn.cursor()
            
            query = "SELECT * FROM open_coded_values WHERE 1=1"
            params = []
            
            if section_id is not None:
                query += " AND section_id = ?"
                params.append(section_id)
            
            if coder_name:
                query += " AND coder_name = ?"
                params.append(coder_name)
            
            query += " ORDER BY confidence_score DESC"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            open_values = []
            for row in rows:
                open_value = OpenCodedValue(
                    open_value_id=row[0],
                    section_id=row[1],
                    value_name=row[2],
                    suggested_category=row[3],
                    confidence_score=row[4],
                    rationale=row[5],
                    coded_date=datetime.fromisoformat(row[6]) if row[6] else None,
                    coder_name=row[7],
                    model_version=row[8],
                    is_validated=bool(row[9])
                )
                open_values.append(open_value)
            
            return open_values
            
        except Exception as e:
            logger.error(f"Error retrieving open coded values: {e}")
            raise DatabaseError(f"Failed to retrieve open coded values: {e}")
    
    async def create_taxonomy_mappings(self, section_id: int) -> List[TaxonomyMapping]:
        """Create mappings between open-coded values and taxonomy values for a section"""
        try:
            # Get open-coded values for this section
            open_values = self.get_open_coded_values(section_id=section_id)
            if not open_values:
                return []
            
            # Get section text for context
            section = self._get_section_by_id(section_id)
            if not section:
                raise TaxonomyValidationException(f"Section {section_id} not found")
            
            # Format open values and taxonomy for AI prompt
            open_values_text = self._format_open_values_for_prompt(open_values)
            taxonomy_text = self._format_taxonomy_for_prompt()
            
            # Create validation prompt
            prompt = CLAUDE_PROMPTS["taxonomy_validation"].format(
                open_coded_values=open_values_text,
                taxonomy_values=taxonomy_text,
                text_section=section.section_text
            )
            
            # Get AI response
            response = await self._get_ai_completion(prompt)
            
            # Parse response
            try:
                result = json.loads(response)
            except json.JSONDecodeError:
                import json_repair
                repaired = json_repair.repair_json(response)
                result = json.loads(repaired)
            
            # Process mappings
            mappings = []
            for mapping_data in result.get('mappings', []):
                # Find the open value
                open_value = next(
                    (ov for ov in open_values if ov.value_name == mapping_data.get('open_value')), 
                    None
                )
                if not open_value:
                    continue
                
                # Get taxonomy value ID if there's a match
                taxonomy_value_id = None
                if mapping_data.get('taxonomy_match'):
                    taxonomy_value_id = self._get_taxonomy_value_id(mapping_data['taxonomy_match'])
                
                mapping = TaxonomyMapping(
                    open_value_id=open_value.open_value_id,
                    taxonomy_value_id=taxonomy_value_id,
                    mapping_type=MappingType(mapping_data.get('mapping_type', 'no_match')),
                    confidence_score=float(mapping_data.get('confidence', 0.0)),
                    rationale=mapping_data.get('rationale', ''),
                    created_date=datetime.now(),
                    created_by=self.app.settings.get('codername', 'AI'),
                    open_value_name=open_value.value_name,
                    taxonomy_value_name=mapping_data.get('taxonomy_match', '')
                )
                
                # Save mapping
                saved_mapping = self.save_taxonomy_mapping(mapping)
                mappings.append(saved_mapping)
            
            # Process taxonomy recommendations
            for rec_data in result.get('taxonomy_recommendations', []):
                recommendation = TaxonomyRecommendation(
                    recommendation_type=RecommendationType(rec_data.get('recommendation_type', 'add_value')),
                    rationale=rec_data.get('details', ''),
                    created_date=datetime.now(),
                    created_by=self.app.settings.get('codername', 'AI')
                )
                self.save_taxonomy_recommendation(recommendation)
            
            logger.info(f"Created {len(mappings)} taxonomy mappings for section {section_id}")
            return mappings
            
        except Exception as e:
            logger.error(f"Error creating taxonomy mappings: {e}")
            raise TaxonomyValidationException(f"Failed to create taxonomy mappings: {e}")
    
    def save_taxonomy_mapping(self, mapping: TaxonomyMapping) -> TaxonomyMapping:
        """Save a taxonomy mapping to the database"""
        try:
            cursor = self.conn.cursor()
            
            cursor.execute("""
                INSERT INTO taxonomy_mappings 
                (open_value_id, taxonomy_value_id, mapping_type, confidence_score, rationale,
                 created_date, created_by)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                mapping.open_value_id, mapping.taxonomy_value_id, mapping.mapping_type.value,
                mapping.confidence_score, mapping.rationale,
                mapping.created_date.isoformat() if mapping.created_date else None,
                mapping.created_by
            ))
            
            mapping.mapping_id = cursor.lastrowid
            self.conn.commit()
            
            logger.info(f"Saved taxonomy mapping for open value {mapping.open_value_id}")
            return mapping
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error saving taxonomy mapping: {e}")
            raise DatabaseError(f"Failed to save taxonomy mapping: {e}")
    
    def save_taxonomy_recommendation(self, recommendation: TaxonomyRecommendation) -> TaxonomyRecommendation:
        """Save a taxonomy recommendation to the database"""
        try:
            cursor = self.conn.cursor()
            
            cursor.execute("""
                INSERT INTO taxonomy_recommendations 
                (recommendation_type, current_value_name, suggested_value_name, suggested_category,
                 rationale, supporting_evidence, frequency_count, created_date, created_by, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                recommendation.recommendation_type.value, recommendation.current_value_name,
                recommendation.suggested_value_name, recommendation.suggested_category,
                recommendation.rationale, recommendation.supporting_evidence,
                recommendation.frequency_count,
                recommendation.created_date.isoformat() if recommendation.created_date else None,
                recommendation.created_by, recommendation.status
            ))
            
            recommendation.recommendation_id = cursor.lastrowid
            self.conn.commit()
            
            logger.info(f"Saved taxonomy recommendation: {recommendation.recommendation_type.value}")
            return recommendation
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error saving taxonomy recommendation: {e}")
            raise DatabaseError(f"Failed to save taxonomy recommendation: {e}")
    
    async def perform_hybrid_coding(self, section_id: int, section_text: str) -> HybridCodingResult:
        """Perform hybrid coding using both open and taxonomy approaches"""
        try:
            # Format taxonomy for prompt
            taxonomy_text = self._format_taxonomy_for_prompt()
            
            # Create hybrid coding prompt
            prompt = CLAUDE_PROMPTS["hybrid_coding"].format(
                taxonomy_values=taxonomy_text,
                text_section=section_text
            )
            
            # Get AI response
            response = await self._get_ai_completion(prompt)
            
            # Parse response
            try:
                result = json.loads(response)
            except json.JSONDecodeError:
                import json_repair
                repaired = json_repair.repair_json(response)
                result = json.loads(repaired)
            
            # Process open values
            open_values = []
            for ov_data in result.get('open_values', []):
                open_value = OpenCodedValue(
                    section_id=section_id,
                    value_name=ov_data.get('value', ''),
                    confidence_score=float(ov_data.get('confidence', 0.0)),
                    rationale=ov_data.get('rationale', ''),
                    coded_date=datetime.now(),
                    coder_name=self.app.settings.get('codername', 'AI_Hybrid'),
                    model_version=getattr(self.app.ai, 'model_version', 'claude-3-5-sonnet')
                )
                open_values.append(open_value)
            
            # Process taxonomy matches
            taxonomy_matches = []
            for tm_data in result.get('taxonomy_matches', []):
                taxonomy_value_id = self._get_taxonomy_value_id(tm_data.get('taxonomy_value', ''))
                
                mapping = TaxonomyMapping(
                    mapping_type=MappingType.DIRECT_MATCH,
                    taxonomy_value_id=taxonomy_value_id,
                    confidence_score=float(tm_data.get('match_confidence', 0.0)),
                    rationale=tm_data.get('mapping_rationale', ''),
                    created_date=datetime.now(),
                    created_by=self.app.settings.get('codername', 'AI_Hybrid'),
                    open_value_name=tm_data.get('open_value', ''),
                    taxonomy_value_name=tm_data.get('taxonomy_value', '')
                )
                taxonomy_matches.append(mapping)
            
            # Get taxonomy gaps
            taxonomy_gaps = [gap.get('missing_value', '') for gap in result.get('taxonomy_gaps', [])]
            
            # Create result object
            hybrid_result = HybridCodingResult(
                section_id=section_id,
                open_values=open_values,
                taxonomy_matches=taxonomy_matches,
                taxonomy_gaps=taxonomy_gaps,
                confidence_score=sum(ov.confidence_score for ov in open_values) / len(open_values) if open_values else 0.0
            )
            
            # Save to database
            self._save_hybrid_coding_result(hybrid_result)
            
            logger.info(f"Completed hybrid coding for section {section_id}")
            return hybrid_result
            
        except Exception as e:
            logger.error(f"Error performing hybrid coding: {e}")
            raise OpenCodingException(f"Failed to perform hybrid coding: {e}")
    
    def get_open_coding_statistics(self, session_ids: Optional[List[int]] = None) -> OpenCodingStatistics:
        """Get statistics for open coding analysis"""
        try:
            cursor = self.conn.cursor()
            
            # Base query conditions
            where_clause = "WHERE 1=1"
            params = []
            
            if session_ids:
                placeholders = ','.join('?' for _ in session_ids)
                where_clause += f" AND ds.session_id IN ({placeholders})"
                params.extend(session_ids)
            
            # Get basic counts
            cursor.execute(f"""
                SELECT COUNT(*), COUNT(DISTINCT ocv.value_name)
                FROM open_coded_values ocv
                JOIN document_sections ds ON ocv.section_id = ds.section_id
                {where_clause}
            """, params)
            total_open_values, unique_values = cursor.fetchone()
            
            # Get most frequent values
            cursor.execute(f"""
                SELECT ocv.value_name, COUNT(*) as frequency
                FROM open_coded_values ocv
                JOIN document_sections ds ON ocv.section_id = ds.section_id
                {where_clause}
                GROUP BY ocv.value_name
                ORDER BY frequency DESC
                LIMIT 10
            """, params)
            most_frequent_values = dict(cursor.fetchall())
            
            # Get category distribution
            cursor.execute(f"""
                SELECT ocv.suggested_category, COUNT(*) as count
                FROM open_coded_values ocv
                JOIN document_sections ds ON ocv.section_id = ds.section_id
                {where_clause}
                WHERE ocv.suggested_category IS NOT NULL AND ocv.suggested_category != ''
                GROUP BY ocv.suggested_category
                ORDER BY count DESC
            """, params)
            category_distribution = dict(cursor.fetchall())
            
            # Get mapping success rate
            cursor.execute(f"""
                SELECT 
                    COUNT(*) as total_mappings,
                    SUM(CASE WHEN tm.mapping_type IN ('direct_match', 'broader_category') THEN 1 ELSE 0 END) as successful_mappings
                FROM taxonomy_mappings tm
                JOIN open_coded_values ocv ON tm.open_value_id = ocv.open_value_id
                JOIN document_sections ds ON ocv.section_id = ds.section_id
                {where_clause}
            """, params)
            
            result = cursor.fetchone()
            total_mappings, successful_mappings = result if result else (0, 0)
            mapping_success_rate = (successful_mappings / total_mappings * 100) if total_mappings > 0 else 0.0
            
            # Get taxonomy gaps count
            cursor.execute(f"""
                SELECT COUNT(*)
                FROM taxonomy_mappings tm
                JOIN open_coded_values ocv ON tm.open_value_id = ocv.open_value_id
                JOIN document_sections ds ON ocv.section_id = ds.section_id
                {where_clause}
                WHERE tm.mapping_type = 'taxonomy_gap'
            """, params)
            taxonomy_gaps_count = cursor.fetchone()[0]
            
            # Get validation coverage
            cursor.execute(f"""
                SELECT 
                    COUNT(*) as total_values,
                    SUM(CASE WHEN ocv.is_validated = 1 THEN 1 ELSE 0 END) as validated_values
                FROM open_coded_values ocv
                JOIN document_sections ds ON ocv.section_id = ds.section_id
                {where_clause}
            """, params)
            
            result = cursor.fetchone()
            total_values, validated_values = result if result else (0, 0)
            validation_coverage = (validated_values / total_values * 100) if total_values > 0 else 0.0
            
            statistics = OpenCodingStatistics(
                total_open_values=total_open_values,
                unique_values=unique_values,
                most_frequent_values=most_frequent_values,
                category_distribution=category_distribution,
                mapping_success_rate=mapping_success_rate,
                taxonomy_gaps_count=taxonomy_gaps_count,
                validation_coverage=validation_coverage
            )
            
            return statistics
            
        except Exception as e:
            logger.error(f"Error getting open coding statistics: {e}")
            raise DatabaseError(f"Failed to get open coding statistics: {e}")
    
    # Private helper methods
    
    def _get_section_by_id(self, section_id: int) -> Optional[DocumentSection]:
        """Get section by ID"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM document_sections WHERE section_id = ?", (section_id,))
            row = cursor.fetchone()
            
            if row:
                return DocumentSection(
                    section_id=row[0],
                    fid=row[1],
                    session_id=row[2],
                    section_number=row[3],
                    section_text=row[4],
                    start_pos=row[5],
                    end_pos=row[6],
                    created_date=datetime.fromisoformat(row[8]) if row[8] else None
                )
            return None
            
        except Exception as e:
            logger.error(f"Error getting section by ID: {e}")
            return None
    
    def _format_open_values_for_prompt(self, open_values: List[OpenCodedValue]) -> str:
        """Format open values for AI prompt"""
        formatted_values = []
        for ov in open_values:
            formatted_values.append(f"- {ov.value_name} ({ov.confidence_score:.2f}): {ov.rationale}")
        return "\n".join(formatted_values)
    
    def _format_taxonomy_for_prompt(self) -> str:
        """Format taxonomy for AI prompt"""
        formatted_taxonomy = []
        for category, values in CORE_VALUES_TAXONOMY.items():
            formatted_taxonomy.append(f"\n## {category}")
            for value_name, description, definition in values:
                formatted_taxonomy.append(f"- **{value_name}**: {description}")
        return "\n".join(formatted_taxonomy)
    
    def _get_taxonomy_value_id(self, value_name: str) -> Optional[int]:
        """Get taxonomy value ID by name"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT value_id FROM core_values_taxonomy WHERE value_name = ?", (value_name,))
            result = cursor.fetchone()
            return result[0] if result else None
        except Exception:
            return None
    
    def _save_hybrid_coding_result(self, result: HybridCodingResult):
        """Save hybrid coding result to database"""
        try:
            cursor = self.conn.cursor()
            
            cursor.execute("""
                INSERT INTO hybrid_coding_results 
                (section_id, open_values_json, taxonomy_matches_json, taxonomy_gaps_json,
                 confidence_score, created_date, created_by)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                result.section_id,
                json.dumps([ov.to_dict() for ov in result.open_values]),
                json.dumps([tm.to_dict() for tm in result.taxonomy_matches]),
                json.dumps(result.taxonomy_gaps),
                result.confidence_score,
                datetime.now().isoformat(),
                self.app.settings.get('codername', 'AI_Hybrid')
            ))
            
            self.conn.commit()
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error saving hybrid coding result: {e}")
    
    async def _get_ai_completion(self, prompt: str) -> str:
        """Get completion from AI service"""
        try:
            if hasattr(self.app.ai, 'get_completion_async'):
                return await self.app.ai.get_completion_async(prompt)
            elif hasattr(self.app.ai, 'get_completion'):
                return self.app.ai.get_completion(prompt)
            else:
                raise OpenCodingException("No completion method available in AI service")
        except Exception as e:
            raise OpenCodingException(f"AI completion failed: {e}")


def create_open_coding_service(app) -> OpenCodingService:
    """Factory function to create OpenCodingService instance"""
    return OpenCodingService(app)