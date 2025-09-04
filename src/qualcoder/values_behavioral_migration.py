"""
Values and Behavioral Enactment Coder - Database Migration Module
This module handles the database migration to add two-stage values and behavioral coding support
to existing QualCoder projects.
"""

import sqlite3
import datetime
import logging
import os
import sys

logger = logging.getLogger(__name__)


class ValuesBehavioralMigration:
    """Handles database migration for Values and Behavioral Enactment Coder"""
    
    MIGRATION_VERSION = "v14_values_behavioral"
    
    def __init__(self, app):
        """Initialize migration with app instance"""
        self.app = app
        self.conn = app.conn
        
    def check_migration_needed(self):
        """Check if the values behavioral migration is needed"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='core_values_taxonomy'")
            result = cursor.fetchone()
            return result is None
        except Exception as e:
            logger.error(f"Error checking migration status: {e}")
            return True
    
    def run_migration(self):
        """Run the complete migration to add values behavioral coding support"""
        if not self.check_migration_needed():
            logger.info("Values behavioral migration already completed")
            return True
            
        try:
            cursor = self.conn.cursor()
            
            # Begin transaction
            cursor.execute("BEGIN TRANSACTION")
            
            # Create all new tables
            self._create_core_tables(cursor)
            self._insert_default_data(cursor)
            self._create_indexes(cursor)
            
            # Update database version in project table
            cursor.execute("UPDATE project SET databaseversion = ? WHERE rowid = 1", (self.MIGRATION_VERSION,))
            
            # Commit transaction
            self.conn.commit()
            
            logger.info("Values behavioral migration completed successfully")
            return True
            
        except Exception as e:
            # Rollback on error
            self.conn.rollback()
            logger.error(f"Migration failed: {e}")
            return False
    
    def _create_core_tables(self, cursor):
        """Create all core tables for values and behavioral coding"""
        
        # Core Values Taxonomy Table
        cursor.execute("""
            CREATE TABLE core_values_taxonomy (
                value_id INTEGER PRIMARY KEY,
                value_name TEXT NOT NULL UNIQUE,
                value_category TEXT NOT NULL,
                description TEXT,
                definition TEXT,
                created_date TEXT DEFAULT (datetime('now')),
                is_active INTEGER DEFAULT 1
            )
        """)
        
        # Values Coding Sessions Table
        cursor.execute("""
            CREATE TABLE values_coding_session (
                session_id INTEGER PRIMARY KEY,
                fid INTEGER NOT NULL,
                coder_name TEXT NOT NULL,
                stage INTEGER NOT NULL CHECK (stage IN (1, 2)),
                session_start TEXT DEFAULT (datetime('now')),
                session_end TEXT,
                status TEXT DEFAULT 'in_progress' CHECK (status IN ('in_progress', 'completed', 'paused')),
                notes TEXT,
                FOREIGN KEY (fid) REFERENCES source(id) ON DELETE CASCADE
            )
        """)
        
        # Document Sections Table
        cursor.execute("""
            CREATE TABLE document_sections (
                section_id INTEGER PRIMARY KEY,
                fid INTEGER NOT NULL,
                session_id INTEGER NOT NULL,
                section_number INTEGER NOT NULL,
                section_text TEXT NOT NULL,
                start_pos INTEGER NOT NULL,
                end_pos INTEGER NOT NULL,
                section_type TEXT DEFAULT 'paragraph',
                created_date TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (fid) REFERENCES source(id) ON DELETE CASCADE,
                FOREIGN KEY (session_id) REFERENCES values_coding_session(session_id) ON DELETE CASCADE,
                UNIQUE(fid, session_id, section_number)
            )
        """)
        
        # Claude Values Suggestions Table
        cursor.execute("""
            CREATE TABLE claude_values_suggestions (
                suggestion_id INTEGER PRIMARY KEY,
                section_id INTEGER NOT NULL,
                value_id INTEGER NOT NULL,
                confidence_score REAL NOT NULL CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
                rationale TEXT,
                suggested_date TEXT DEFAULT (datetime('now')),
                model_version TEXT,
                FOREIGN KEY (section_id) REFERENCES document_sections(section_id) ON DELETE CASCADE,
                FOREIGN KEY (value_id) REFERENCES core_values_taxonomy(value_id) ON DELETE CASCADE
            )
        """)
        
        # Values Coding Table (Stage 1)
        cursor.execute("""
            CREATE TABLE values_coding (
                coding_id INTEGER PRIMARY KEY,
                section_id INTEGER NOT NULL,
                value_id INTEGER,
                custom_value_name TEXT,
                is_manual_entry INTEGER DEFAULT 0,
                selected_from_suggestion INTEGER DEFAULT 0,
                confidence_level TEXT CHECK (confidence_level IN ('high', 'medium', 'low')),
                coder_notes TEXT,
                coded_date TEXT DEFAULT (datetime('now')),
                coder_name TEXT NOT NULL,
                locked_date TEXT,
                is_locked INTEGER DEFAULT 0,
                FOREIGN KEY (section_id) REFERENCES document_sections(section_id) ON DELETE CASCADE,
                FOREIGN KEY (value_id) REFERENCES core_values_taxonomy(value_id) ON DELETE SET NULL,
                UNIQUE(section_id, coder_name)
            )
        """)
        
        # Document Sentences Table
        cursor.execute("""
            CREATE TABLE document_sentences (
                sentence_id INTEGER PRIMARY KEY,
                section_id INTEGER NOT NULL,
                sentence_number INTEGER NOT NULL,
                sentence_text TEXT NOT NULL,
                start_pos INTEGER NOT NULL,
                end_pos INTEGER NOT NULL,
                created_date TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (section_id) REFERENCES document_sections(section_id) ON DELETE CASCADE,
                UNIQUE(section_id, sentence_number)
            )
        """)
        
        # Claude Behavioral Suggestions Table
        cursor.execute("""
            CREATE TABLE claude_behavioral_suggestions (
                suggestion_id INTEGER PRIMARY KEY,
                sentence_id INTEGER NOT NULL,
                behavioral_score INTEGER NOT NULL CHECK (behavioral_score BETWEEN -3 AND 3),
                confidence_score REAL NOT NULL CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
                rationale TEXT,
                suggested_date TEXT DEFAULT (datetime('now')),
                model_version TEXT,
                FOREIGN KEY (sentence_id) REFERENCES document_sentences(sentence_id) ON DELETE CASCADE
            )
        """)
        
        # Behavioral Coding Table (Stage 2)
        cursor.execute("""
            CREATE TABLE behavioral_coding (
                coding_id INTEGER PRIMARY KEY,
                sentence_id INTEGER NOT NULL,
                values_coding_id INTEGER NOT NULL,
                behavioral_score INTEGER NOT NULL CHECK (behavioral_score BETWEEN -3 AND 3),
                selected_from_suggestion INTEGER DEFAULT 0,
                confidence_level TEXT CHECK (confidence_level IN ('high', 'medium', 'low')),
                coder_rationale TEXT,
                coded_date TEXT DEFAULT (datetime('now')),
                coder_name TEXT NOT NULL,
                locked_date TEXT,
                is_locked INTEGER DEFAULT 0,
                FOREIGN KEY (sentence_id) REFERENCES document_sentences(sentence_id) ON DELETE CASCADE,
                FOREIGN KEY (values_coding_id) REFERENCES values_coding(coding_id) ON DELETE CASCADE,
                UNIQUE(sentence_id, coder_name)
            )
        """)
        
        # Behavioral Scale Definitions Table
        cursor.execute("""
            CREATE TABLE behavioral_scale_definitions (
                scale_point INTEGER PRIMARY KEY CHECK (scale_point BETWEEN -3 AND 3),
                scale_name TEXT NOT NULL,
                short_description TEXT NOT NULL,
                full_description TEXT NOT NULL,
                examples TEXT,
                created_date TEXT DEFAULT (datetime('now'))
            )
        """)
        
        # Coding Progress Tracking Table
        cursor.execute("""
            CREATE TABLE coding_progress (
                progress_id INTEGER PRIMARY KEY,
                session_id INTEGER NOT NULL,
                fid INTEGER NOT NULL,
                total_sections INTEGER DEFAULT 0,
                sections_values_coded INTEGER DEFAULT 0,
                sections_values_locked INTEGER DEFAULT 0,
                total_sentences INTEGER DEFAULT 0,
                sentences_behavioral_coded INTEGER DEFAULT 0,
                sentences_behavioral_locked INTEGER DEFAULT 0,
                stage_1_complete INTEGER DEFAULT 0,
                stage_2_complete INTEGER DEFAULT 0,
                last_updated TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (session_id) REFERENCES values_coding_session(session_id) ON DELETE CASCADE,
                FOREIGN KEY (fid) REFERENCES source(id) ON DELETE CASCADE,
                UNIQUE(session_id, fid)
            )
        """)
        
        # Export Templates Table
        cursor.execute("""
            CREATE TABLE export_templates (
                template_id INTEGER PRIMARY KEY,
                template_name TEXT NOT NULL UNIQUE,
                template_type TEXT NOT NULL CHECK (template_type IN ('values_summary', 'behavioral_summary', 'combined', 'detailed')),
                export_format TEXT NOT NULL CHECK (export_format IN ('csv', 'json', 'excel', 'pdf')),
                template_config TEXT,
                created_date TEXT DEFAULT (datetime('now')),
                created_by TEXT,
                is_default INTEGER DEFAULT 0
            )
        """)
        
    def _insert_default_data(self, cursor):
        """Insert default data for values taxonomy and behavioral scale definitions"""
        
        # Insert core values taxonomy
        values_data = [
            # Achievement Values
            ('Achievement', 'Achievement', 'Personal accomplishment and success', 'The drive to accomplish personal goals and demonstrate competence'),
            ('Excellence', 'Achievement', 'Pursuing the highest quality and standards', 'Commitment to superior performance and continuous improvement'),
            ('Recognition', 'Achievement', 'Seeking acknowledgment and appreciation', 'Desire for appreciation, respect, and acknowledgment from others'),
            ('Competence', 'Achievement', 'Developing and demonstrating skills', 'Building and showcasing professional and personal capabilities'),
            
            # Benevolence Values  
            ('Benevolence', 'Benevolence', 'Concern for the welfare of others', 'Acting with kindness and generosity toward others'),
            ('Service', 'Benevolence', 'Helping and supporting others', 'Dedication to meeting the needs of others and contributing to their wellbeing'),
            ('Compassion', 'Benevolence', 'Empathy and care for others', 'Deep concern for the suffering and needs of others'),
            ('Generosity', 'Benevolence', 'Giving freely to others', 'Willingness to share resources, time, and effort with others'),
            
            # Conformity Values
            ('Conformity', 'Conformity', 'Following rules and social expectations', 'Adherence to social norms, rules, and expectations'),
            ('Tradition', 'Conformity', 'Respecting customs and established practices', 'Commitment to maintaining cultural and organizational traditions'),
            ('Obedience', 'Conformity', 'Following authority and instructions', 'Compliance with legitimate authority and established procedures'),
            ('Politeness', 'Conformity', 'Social courtesy and respect', 'Maintaining proper social behavior and showing respect to others'),
            
            # Hedonism Values
            ('Hedonism', 'Hedonism', 'Seeking pleasure and enjoyment', 'Pursuit of personal pleasure, enjoyment, and gratification'),
            ('Fun', 'Hedonism', 'Engaging in enjoyable activities', 'Seeking entertainment, amusement, and lighthearted experiences'),
            ('Comfort', 'Hedonism', 'Seeking ease and convenience', 'Preference for comfortable, convenient, and stress-free conditions'),
            
            # Power Values
            ('Power', 'Power', 'Seeking control and influence', 'Desire for control, dominance, and influence over others and situations'),
            ('Authority', 'Power', 'Exercising leadership and control', 'Seeking positions of authority and decision-making responsibility'),
            ('Dominance', 'Power', 'Controlling others and situations', 'Desire to control and direct others behavior and outcomes'),
            ('Status', 'Power', 'Seeking high social position', 'Pursuit of prestige, social rank, and elevated social position'),
            
            # Security Values
            ('Security', 'Security', 'Seeking safety and stability', 'Desire for safety, stability, and predictability in life and work'),
            ('Safety', 'Security', 'Protecting from harm and danger', 'Ensuring protection from physical, emotional, and financial risks'),
            ('Stability', 'Security', 'Maintaining consistent conditions', 'Preference for predictable, steady, and unchanging circumstances'),
            ('Order', 'Security', 'Organization and systematic approach', 'Maintaining structure, organization, and systematic processes'),
            
            # Self-Direction Values
            ('Self-Direction', 'Self-Direction', 'Independence and autonomy', 'Acting independently and making autonomous decisions'),
            ('Independence', 'Self-Direction', 'Freedom from external control', 'Operating without external constraints or supervision'),
            ('Autonomy', 'Self-Direction', 'Self-governance and choice', 'Having the freedom to make ones own choices and decisions'),
            ('Creativity', 'Self-Direction', 'Innovation and original thinking', 'Developing new ideas, approaches, and creative solutions'),
            
            # Stimulation Values
            ('Stimulation', 'Stimulation', 'Seeking excitement and novelty', 'Pursuit of excitement, novelty, and challenging experiences'),
            ('Adventure', 'Stimulation', 'Seeking new and exciting experiences', 'Pursuing novel, thrilling, and adventurous activities'),
            ('Variety', 'Stimulation', 'Seeking diverse experiences', 'Preference for diverse, varied, and changing experiences'),
            ('Challenge', 'Stimulation', 'Embracing difficult tasks', 'Seeking difficult, demanding, and challenging work or situations'),
            
            # Universalism Values
            ('Universalism', 'Universalism', 'Concern for all people and nature', 'Care and concern for the welfare of all people and the natural world'),
            ('Justice', 'Universalism', 'Fairness and equality', 'Commitment to fairness, equality, and just treatment for all'),
            ('Equality', 'Universalism', 'Equal treatment and opportunities', 'Belief that all people deserve equal treatment and opportunities'),
            ('Environmental', 'Universalism', 'Protecting the natural world', 'Commitment to environmental protection and sustainability'),
            
            # Additional Core Values
            ('Integrity', 'Core', 'Honesty and moral consistency', 'Acting in accordance with moral and ethical principles consistently'),
            ('Respect', 'Core', 'Valuing others dignity and worth', 'Treating others with dignity, consideration, and appreciation'),
            ('Responsibility', 'Core', 'Accountability for actions and duties', 'Taking ownership of actions, decisions, and their consequences'),
            ('Trust', 'Core', 'Reliability and dependability', 'Being trustworthy and having confidence in others trustworthiness')
        ]
        
        cursor.executemany("""
            INSERT INTO core_values_taxonomy (value_name, value_category, description, definition) 
            VALUES (?, ?, ?, ?)
        """, values_data)
        
        # Insert behavioral scale definitions
        scale_data = [
            (-3, 'Extraordinary Violation', 'Systematic undermining of values', 
             'Deliberately and systematically acting in ways that undermine or contradict the core value, often with significant negative consequences', 
             'Deliberately sabotaging team efforts, systematically violating ethical standards, actively working against organizational values'),
            (-2, 'Active Violation', 'Deliberate contradiction of values', 
             'Consciously and deliberately acting in ways that directly contradict or oppose the core value', 
             'Knowingly breaking rules for personal gain, deliberately treating others unfairly, actively resisting positive changes'),
            (-1, 'Capitulating', 'Surrender through inaction', 
             'Failing to act in alignment with values when action was possible, essentially surrendering to opposing forces through passivity', 
             'Remaining silent when speaking up is needed, avoiding responsibility when leadership is required, giving up on important principles under pressure'),
            (0, 'Indifference', 'Apathetic disengagement', 
             'Showing no particular care or concern for the value, neither supporting nor opposing it, remaining neutral or disengaged', 
             'Being indifferent to team success or failure, showing no concern for ethical issues, remaining uninvolved in important decisions'),
            (1, 'Compromising', 'Partial selective enactment', 
             'Acting in alignment with the value only partially or selectively, making compromises that weaken full commitment', 
             'Following some rules but not others, helping some people but not others, pursuing quality only when convenient'),
            (2, 'Active Enacting', 'Consistent deliberate alignment', 
             'Consistently and deliberately acting in ways that align with and support the core value, making it a priority in decisions and actions', 
             'Consistently following through on commitments, regularly helping team members, maintaining high standards even under pressure'),
            (3, 'Extraordinary Enacting', 'Exceptional commitment with sacrifice', 
             'Going above and beyond to enact the value, often at significant personal or professional cost, demonstrating exceptional commitment', 
             'Sacrificing personal advancement to maintain integrity, taking significant risks to help others, persevering through extreme challenges to uphold principles')
        ]
        
        cursor.executemany("""
            INSERT INTO behavioral_scale_definitions (scale_point, scale_name, short_description, full_description, examples) 
            VALUES (?, ?, ?, ?, ?)
        """, scale_data)
        
    def _create_indexes(self, cursor):
        """Create indexes for performance optimization"""
        
        indexes = [
            "CREATE INDEX idx_values_coding_session_fid ON values_coding_session(fid)",
            "CREATE INDEX idx_values_coding_session_stage ON values_coding_session(stage)",
            "CREATE INDEX idx_document_sections_fid ON document_sections(fid)",
            "CREATE INDEX idx_document_sections_session ON document_sections(session_id)",
            "CREATE INDEX idx_claude_values_suggestions_section ON claude_values_suggestions(section_id)",
            "CREATE INDEX idx_claude_values_suggestions_value ON claude_values_suggestions(value_id)",
            "CREATE INDEX idx_values_coding_section ON values_coding(section_id)",
            "CREATE INDEX idx_values_coding_value ON values_coding(value_id)",
            "CREATE INDEX idx_document_sentences_section ON document_sentences(section_id)",
            "CREATE INDEX idx_claude_behavioral_suggestions_sentence ON claude_behavioral_suggestions(sentence_id)",
            "CREATE INDEX idx_behavioral_coding_sentence ON behavioral_coding(sentence_id)",
            "CREATE INDEX idx_behavioral_coding_values ON behavioral_coding(values_coding_id)",
            "CREATE INDEX idx_coding_progress_session ON coding_progress(session_id)",
            "CREATE INDEX idx_coding_progress_fid ON coding_progress(fid)"
        ]
        
        for index in indexes:
            cursor.execute(index)


def migrate_project_to_values_behavioral(app):
    """Convenience function to migrate a project to support values behavioral coding"""
    migration = ValuesBehavioralMigration(app)
    return migration.run_migration()