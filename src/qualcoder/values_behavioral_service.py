"""
Values and Behavioral Enactment Coder - Service Layer
Contains service classes for database operations and business logic
"""

import sqlite3
import json
import re
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
import logging

from .values_behavioral_models import (
    CoreValue, ValuesCodingSession, DocumentSection, ClaudeValuesSuggestion,
    ValuesCoding, DocumentSentence, ClaudeBehavioralSuggestion, BehavioralCoding,
    BehavioralScaleDefinition, CodingProgress, ExportTemplate, CodingStatistics,
    ValidationError, DatabaseError, ClaudeAPIError
)
from .values_behavioral_constants import (
    CodingStage, SessionStatus, ConfidenceLevel, SectionType,
    BEHAVIORAL_SCALE, CLAUDE_PROMPTS, CONFIDENCE_THRESHOLDS
)

logger = logging.getLogger(__name__)


class ValuesBehavioralService:
    """Main service class for values and behavioral coding operations"""
    
    def __init__(self, app):
        """Initialize with app instance"""
        self.app = app
        self.conn = app.conn
        
    def get_core_values(self, active_only: bool = True) -> List[CoreValue]:
        """Retrieve core values from taxonomy"""
        try:
            cursor = self.conn.cursor()
            query = "SELECT * FROM core_values_taxonomy"
            if active_only:
                query += " WHERE is_active = 1"
            query += " ORDER BY value_category, value_name"
            
            cursor.execute(query)
            rows = cursor.fetchall()
            
            values = []
            for row in rows:
                value = CoreValue(
                    value_id=row[0],
                    value_name=row[1],
                    value_category=row[2],
                    description=row[3],
                    definition=row[4],
                    created_date=datetime.fromisoformat(row[5]) if row[5] else None,
                    is_active=bool(row[6])
                )
                values.append(value)
            
            return values
        except Exception as e:
            logger.error(f"Error retrieving core values: {e}")
            raise DatabaseError(f"Failed to retrieve core values: {e}")
    
    def create_coding_session(self, fid: int, coder_name: str, stage: CodingStage = CodingStage.VALUES, notes: str = "") -> ValuesCodingSession:
        """Create a new coding session"""
        try:
            cursor = self.conn.cursor()
            
            session = ValuesCodingSession(
                fid=fid,
                coder_name=coder_name,
                stage=stage,
                session_start=datetime.now(),
                status=SessionStatus.IN_PROGRESS,
                notes=notes
            )
            
            cursor.execute("""
                INSERT INTO values_coding_session (fid, coder_name, stage, session_start, status, notes)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (session.fid, session.coder_name, session.stage.value, 
                  session.session_start.isoformat(), session.status.value, session.notes))
            
            session.session_id = cursor.lastrowid
            self.conn.commit()
            
            # Create progress tracking entry
            self._create_progress_tracking(session.session_id, fid)
            
            logger.info(f"Created coding session {session.session_id} for file {fid}")
            return session
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error creating coding session: {e}")
            raise DatabaseError(f"Failed to create coding session: {e}")
    
    def get_coding_sessions(self, fid: Optional[int] = None, coder_name: Optional[str] = None) -> List[ValuesCodingSession]:
        """Retrieve coding sessions"""
        try:
            cursor = self.conn.cursor()
            
            query = "SELECT * FROM values_coding_session WHERE 1=1"
            params = []
            
            if fid is not None:
                query += " AND fid = ?"
                params.append(fid)
            
            if coder_name:
                query += " AND coder_name = ?"
                params.append(coder_name)
            
            query += " ORDER BY session_start DESC"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            sessions = []
            for row in rows:
                session = ValuesCodingSession(
                    session_id=row[0],
                    fid=row[1],
                    coder_name=row[2],
                    stage=CodingStage(row[3]),
                    session_start=datetime.fromisoformat(row[4]) if row[4] else None,
                    session_end=datetime.fromisoformat(row[5]) if row[5] else None,
                    status=SessionStatus(row[6]),
                    notes=row[7] if row[7] else ""
                )
                sessions.append(session)
            
            return sessions
            
        except Exception as e:
            logger.error(f"Error retrieving coding sessions: {e}")
            raise DatabaseError(f"Failed to retrieve coding sessions: {e}")
    
    def parse_document_sections(self, fid: int, session_id: int, full_text: str, use_ai: bool = True) -> List[DocumentSection]:
        """Parse document into sections for values coding"""
        try:
            if use_ai and hasattr(self.app, 'ai') and self.app.ai.is_ready():
                sections = self._ai_parse_sections(full_text)
            else:
                sections = self._rule_based_parse_sections(full_text)
            
            # Save sections to database
            cursor = self.conn.cursor()
            saved_sections = []
            
            for i, section_data in enumerate(sections):
                section = DocumentSection(
                    fid=fid,
                    session_id=session_id,
                    section_number=i + 1,
                    section_text=section_data['text'],
                    start_pos=section_data['start_pos'],
                    end_pos=section_data['end_pos'],
                    section_type=SectionType.PARAGRAPH,
                    created_date=datetime.now()
                )
                
                cursor.execute("""
                    INSERT INTO document_sections (fid, session_id, section_number, section_text, 
                                                 start_pos, end_pos, section_type, created_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (section.fid, section.session_id, section.section_number, section.section_text,
                      section.start_pos, section.end_pos, section.section_type.value, 
                      section.created_date.isoformat()))
                
                section.section_id = cursor.lastrowid
                saved_sections.append(section)
            
            self.conn.commit()
            
            # Update progress tracking
            self._update_progress_sections(session_id, len(saved_sections))
            
            logger.info(f"Parsed {len(saved_sections)} sections for session {session_id}")
            return saved_sections
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error parsing document sections: {e}")
            raise DatabaseError(f"Failed to parse document sections: {e}")
    
    def get_claude_values_suggestions(self, section_id: int, section_text: str, use_cache: bool = True) -> List[ClaudeValuesSuggestion]:
        """Get Claude AI suggestions for values coding"""
        try:
            # Check cache first if requested
            if use_cache:
                cached_suggestions = self._get_cached_values_suggestions(section_id)
                if cached_suggestions:
                    return cached_suggestions
            
            # Get AI suggestions
            if not hasattr(self.app, 'ai') or not self.app.ai.is_ready():
                raise ClaudeAPIError("AI service not available")
            
            # Prepare values list for prompt
            core_values = self.get_core_values()
            values_list = [f"- {v.value_name}: {v.description}" for v in core_values]
            values_text = "\n".join(values_list)
            
            # Create prompt
            prompt = CLAUDE_PROMPTS["values_identification"].format(
                values_list=values_text,
                text_section=section_text
            )
            
            # Get AI response
            response = self.app.ai.get_completion(prompt)
            suggestions_data = json.loads(response)
            
            # Save suggestions to database
            cursor = self.conn.cursor()
            suggestions = []
            
            for suggestion_data in suggestions_data.get('values', []):
                # Find value_id by name
                value_id = self._get_value_id_by_name(suggestion_data['value'])
                if not value_id:
                    continue
                
                suggestion = ClaudeValuesSuggestion(
                    section_id=section_id,
                    value_id=value_id,
                    confidence_score=suggestion_data.get('confidence', 0.0),
                    rationale=suggestion_data.get('rationale', ''),
                    suggested_date=datetime.now(),
                    model_version=getattr(self.app.ai, 'model_version', 'unknown')
                )
                
                cursor.execute("""
                    INSERT INTO claude_values_suggestions (section_id, value_id, confidence_score, 
                                                          rationale, suggested_date, model_version)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (suggestion.section_id, suggestion.value_id, suggestion.confidence_score,
                      suggestion.rationale, suggestion.suggested_date.isoformat(), suggestion.model_version))
                
                suggestion.suggestion_id = cursor.lastrowid
                suggestion.value_name = suggestion_data['value']
                suggestions.append(suggestion)
            
            self.conn.commit()
            logger.info(f"Generated {len(suggestions)} values suggestions for section {section_id}")
            return suggestions
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing Claude response: {e}")
            raise ClaudeAPIError(f"Invalid response from Claude API: {e}")
        except Exception as e:
            logger.error(f"Error getting values suggestions: {e}")
            raise ClaudeAPIError(f"Failed to get values suggestions: {e}")
    
    def save_values_coding(self, section_id: int, value_id: Optional[int], custom_value_name: str, 
                          coder_name: str, confidence_level: ConfidenceLevel, coder_notes: str,
                          selected_from_suggestion: bool = False) -> ValuesCoding:
        """Save human-validated values coding"""
        try:
            cursor = self.conn.cursor()
            
            # Validate inputs
            if not value_id and not custom_value_name:
                raise ValidationError("Either value_id or custom_value_name must be provided")
            
            coding = ValuesCoding(
                section_id=section_id,
                value_id=value_id,
                custom_value_name=custom_value_name,
                is_manual_entry=(custom_value_name != ""),
                selected_from_suggestion=selected_from_suggestion,
                confidence_level=confidence_level,
                coder_notes=coder_notes,
                coded_date=datetime.now(),
                coder_name=coder_name,
                is_locked=False
            )
            
            # Use INSERT OR REPLACE to handle updates
            cursor.execute("""
                INSERT OR REPLACE INTO values_coding 
                (section_id, value_id, custom_value_name, is_manual_entry, selected_from_suggestion,
                 confidence_level, coder_notes, coded_date, coder_name, is_locked)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (coding.section_id, coding.value_id, coding.custom_value_name, 
                  coding.is_manual_entry, coding.selected_from_suggestion,
                  coding.confidence_level.value if coding.confidence_level else None,
                  coding.coder_notes, coding.coded_date.isoformat(), coding.coder_name, coding.is_locked))
            
            if not coding.coding_id:
                coding.coding_id = cursor.lastrowid
            
            self.conn.commit()
            
            # Update progress tracking
            session_id = self._get_session_id_for_section(section_id)
            self._update_progress_values_coded(session_id)
            
            logger.info(f"Saved values coding {coding.coding_id} for section {section_id}")
            return coding
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error saving values coding: {e}")
            raise DatabaseError(f"Failed to save values coding: {e}")
    
    def lock_values_coding(self, coding_id: int, coder_name: str) -> bool:
        """Lock values coding to prevent further modifications"""
        try:
            cursor = self.conn.cursor()
            
            cursor.execute("""
                UPDATE values_coding 
                SET is_locked = 1, locked_date = ?
                WHERE coding_id = ? AND coder_name = ? AND is_locked = 0
            """, (datetime.now().isoformat(), coding_id, coder_name))
            
            if cursor.rowcount == 0:
                raise ValidationError("Coding not found or already locked")
            
            self.conn.commit()
            logger.info(f"Locked values coding {coding_id}")
            return True
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error locking values coding: {e}")
            raise DatabaseError(f"Failed to lock values coding: {e}")
    
    def parse_section_sentences(self, section_id: int, section_text: str, use_ai: bool = True) -> List[DocumentSentence]:
        """Parse section into sentences for behavioral coding"""
        try:
            if use_ai and hasattr(self.app, 'ai') and self.app.ai.is_ready():
                sentences = self._ai_parse_sentences(section_text)
            else:
                sentences = self._rule_based_parse_sentences(section_text)
            
            # Save sentences to database
            cursor = self.conn.cursor()
            saved_sentences = []
            
            for i, sentence_data in enumerate(sentences):
                sentence = DocumentSentence(
                    section_id=section_id,
                    sentence_number=i + 1,
                    sentence_text=sentence_data['text'],
                    start_pos=sentence_data['start_pos'],
                    end_pos=sentence_data['end_pos'],
                    created_date=datetime.now()
                )
                
                cursor.execute("""
                    INSERT INTO document_sentences (section_id, sentence_number, sentence_text,
                                                   start_pos, end_pos, created_date)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (sentence.section_id, sentence.sentence_number, sentence.sentence_text,
                      sentence.start_pos, sentence.end_pos, sentence.created_date.isoformat()))
                
                sentence.sentence_id = cursor.lastrowid
                saved_sentences.append(sentence)
            
            self.conn.commit()
            
            # Update progress tracking
            session_id = self._get_session_id_for_section(section_id)
            self._update_progress_sentences(session_id, len(saved_sentences))
            
            logger.info(f"Parsed {len(saved_sentences)} sentences for section {section_id}")
            return saved_sentences
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error parsing sentences: {e}")
            raise DatabaseError(f"Failed to parse sentences: {e}")
    
    def get_behavioral_scale_definitions(self) -> List[BehavioralScaleDefinition]:
        """Get behavioral scale definitions"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM behavioral_scale_definitions ORDER BY scale_point")
            rows = cursor.fetchall()
            
            definitions = []
            for row in rows:
                definition = BehavioralScaleDefinition(
                    scale_point=row[0],
                    scale_name=row[1],
                    short_description=row[2],
                    full_description=row[3],
                    examples=row[4],
                    created_date=datetime.fromisoformat(row[5]) if row[5] else None
                )
                definitions.append(definition)
            
            return definitions
            
        except Exception as e:
            logger.error(f"Error retrieving behavioral scale definitions: {e}")
            raise DatabaseError(f"Failed to retrieve behavioral scale definitions: {e}")
    
    def get_coding_progress(self, session_id: int) -> Optional[CodingProgress]:
        """Get coding progress for a session"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM coding_progress WHERE session_id = ?", (session_id,))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            progress = CodingProgress(
                progress_id=row[0],
                session_id=row[1],
                fid=row[2],
                total_sections=row[3],
                sections_values_coded=row[4],
                sections_values_locked=row[5],
                total_sentences=row[6],
                sentences_behavioral_coded=row[7],
                sentences_behavioral_locked=row[8],
                stage_1_complete=bool(row[9]),
                stage_2_complete=bool(row[10]),
                last_updated=datetime.fromisoformat(row[11]) if row[11] else None
            )
            
            return progress
            
        except Exception as e:
            logger.error(f"Error retrieving coding progress: {e}")
            raise DatabaseError(f"Failed to retrieve coding progress: {e}")
    
    # Private helper methods
    
    def _ai_parse_sections(self, full_text: str) -> List[Dict[str, Any]]:
        """Use AI to parse text into sections"""
        prompt = CLAUDE_PROMPTS["section_parsing"].format(full_text=full_text)
        response = self.app.ai.get_completion(prompt)
        result = json.loads(response)
        
        sections = []
        for section_data in result.get('sections', []):
            sections.append({
                'text': section_data['section_text'],
                'start_pos': section_data['start_pos'],
                'end_pos': section_data['end_pos']
            })
        
        return sections
    
    def _rule_based_parse_sections(self, full_text: str) -> List[Dict[str, Any]]:
        """Use rule-based approach to parse text into sections"""
        # Simple paragraph-based parsing
        paragraphs = re.split(r'\n\s*\n', full_text.strip())
        sections = []
        current_pos = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if len(paragraph) < 20:  # Skip very short paragraphs
                current_pos += len(paragraph) + 2
                continue
            
            start_pos = full_text.find(paragraph, current_pos)
            end_pos = start_pos + len(paragraph)
            
            sections.append({
                'text': paragraph,
                'start_pos': start_pos,
                'end_pos': end_pos
            })
            
            current_pos = end_pos + 1
        
        return sections
    
    def _ai_parse_sentences(self, section_text: str) -> List[Dict[str, Any]]:
        """Use AI to parse section into sentences"""
        prompt = CLAUDE_PROMPTS["sentence_parsing"].format(section_text=section_text)
        response = self.app.ai.get_completion(prompt)
        result = json.loads(response)
        
        sentences = []
        for sentence_data in result.get('sentences', []):
            sentences.append({
                'text': sentence_data['sentence_text'],
                'start_pos': sentence_data['start_pos'],
                'end_pos': sentence_data['end_pos']
            })
        
        return sentences
    
    def _rule_based_parse_sentences(self, section_text: str) -> List[Dict[str, Any]]:
        """Use rule-based approach to parse section into sentences"""
        # Simple sentence boundary detection
        sentence_endings = re.compile(r'[.!?]+\s+')
        sentences = []
        current_pos = 0
        
        for match in sentence_endings.finditer(section_text):
            sentence_text = section_text[current_pos:match.end()].strip()
            if len(sentence_text) > 10:  # Skip very short sentences
                sentences.append({
                    'text': sentence_text,
                    'start_pos': current_pos,
                    'end_pos': match.end()
                })
            current_pos = match.end()
        
        # Handle final sentence if it doesn't end with punctuation
        if current_pos < len(section_text):
            final_sentence = section_text[current_pos:].strip()
            if len(final_sentence) > 10:
                sentences.append({
                    'text': final_sentence,
                    'start_pos': current_pos,
                    'end_pos': len(section_text)
                })
        
        return sentences
    
    def _get_cached_values_suggestions(self, section_id: int) -> List[ClaudeValuesSuggestion]:
        """Get cached values suggestions for a section"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT cvs.*, cvt.value_name 
                FROM claude_values_suggestions cvs
                JOIN core_values_taxonomy cvt ON cvs.value_id = cvt.value_id
                WHERE cvs.section_id = ?
                ORDER BY cvs.confidence_score DESC
            """, (section_id,))
            
            rows = cursor.fetchall()
            suggestions = []
            
            for row in rows:
                suggestion = ClaudeValuesSuggestion(
                    suggestion_id=row[0],
                    section_id=row[1],
                    value_id=row[2],
                    confidence_score=row[3],
                    rationale=row[4],
                    suggested_date=datetime.fromisoformat(row[5]) if row[5] else None,
                    model_version=row[6],
                    value_name=row[7]
                )
                suggestions.append(suggestion)
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Error getting cached suggestions: {e}")
            return []
    
    def _get_value_id_by_name(self, value_name: str) -> Optional[int]:
        """Get value ID by name"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT value_id FROM core_values_taxonomy WHERE value_name = ?", (value_name,))
            result = cursor.fetchone()
            return result[0] if result else None
        except Exception:
            return None
    
    def _get_session_id_for_section(self, section_id: int) -> int:
        """Get session ID for a section"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT session_id FROM document_sections WHERE section_id = ?", (section_id,))
        result = cursor.fetchone()
        return result[0] if result else 0
    
    def _create_progress_tracking(self, session_id: int, fid: int):
        """Create progress tracking entry for a session"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO coding_progress (session_id, fid, last_updated)
            VALUES (?, ?, ?)
        """, (session_id, fid, datetime.now().isoformat()))
    
    def _update_progress_sections(self, session_id: int, total_sections: int):
        """Update progress with total sections"""
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE coding_progress 
            SET total_sections = ?, last_updated = ?
            WHERE session_id = ?
        """, (total_sections, datetime.now().isoformat(), session_id))
    
    def _update_progress_sentences(self, session_id: int, additional_sentences: int):
        """Update progress with additional sentences"""
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE coding_progress 
            SET total_sentences = total_sentences + ?, last_updated = ?
            WHERE session_id = ?
        """, (additional_sentences, datetime.now().isoformat(), session_id))
    
    def _update_progress_values_coded(self, session_id: int):
        """Update progress when values coding is completed"""
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE coding_progress 
            SET sections_values_coded = (
                SELECT COUNT(*) FROM values_coding vc
                JOIN document_sections ds ON vc.section_id = ds.section_id
                WHERE ds.session_id = ?
            ),
            last_updated = ?
            WHERE session_id = ?
        """, (session_id, datetime.now().isoformat(), session_id))