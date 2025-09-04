-- Values and Behavioral Enactment Coder - Database Schema Extensions
-- This file contains the additional tables needed for two-stage values and behavioral coding

-- Core Values Taxonomy Table
-- Stores the 32 core values from Phase 0 research
CREATE TABLE core_values_taxonomy (
    value_id INTEGER PRIMARY KEY,
    value_name TEXT NOT NULL UNIQUE,
    value_category TEXT NOT NULL,
    description TEXT,
    definition TEXT,
    created_date TEXT DEFAULT (datetime('now')),
    is_active INTEGER DEFAULT 1
);

-- Values Coding Sessions Table
-- Tracks two-stage coding sessions for documents
CREATE TABLE values_coding_session (
    session_id INTEGER PRIMARY KEY,
    fid INTEGER NOT NULL,                    -- Reference to source.id (file)
    coder_name TEXT NOT NULL,
    stage INTEGER NOT NULL CHECK (stage IN (1, 2)), -- 1=Values, 2=Behavioral
    session_start TEXT DEFAULT (datetime('now')),
    session_end TEXT,
    status TEXT DEFAULT 'in_progress' CHECK (status IN ('in_progress', 'completed', 'paused')),
    notes TEXT,
    FOREIGN KEY (fid) REFERENCES source(id) ON DELETE CASCADE
);

-- Document Sections Table 
-- Breaks documents into sections for Stage 1 values coding
CREATE TABLE document_sections (
    section_id INTEGER PRIMARY KEY,
    fid INTEGER NOT NULL,                    -- Reference to source.id (file) 
    session_id INTEGER NOT NULL,            -- Reference to coding session
    section_number INTEGER NOT NULL,        -- Sequential section number within document
    section_text TEXT NOT NULL,             -- The actual text content of the section
    start_pos INTEGER NOT NULL,             -- Character position where section starts
    end_pos INTEGER NOT NULL,               -- Character position where section ends
    section_type TEXT DEFAULT 'paragraph',  -- Type: 'paragraph', 'sentence', 'custom'
    created_date TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (fid) REFERENCES source(id) ON DELETE CASCADE,
    FOREIGN KEY (session_id) REFERENCES values_coding_session(session_id) ON DELETE CASCADE,
    UNIQUE(fid, session_id, section_number)
);

-- Claude Values Suggestions Table
-- Stores AI-generated value suggestions for each section
CREATE TABLE claude_values_suggestions (
    suggestion_id INTEGER PRIMARY KEY,
    section_id INTEGER NOT NULL,            -- Reference to document_sections
    value_id INTEGER NOT NULL,              -- Reference to core_values_taxonomy
    confidence_score REAL NOT NULL CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
    rationale TEXT,                         -- Claude's explanation for the suggestion
    suggested_date TEXT DEFAULT (datetime('now')),
    model_version TEXT,                     -- Which Claude model version made the suggestion
    FOREIGN KEY (section_id) REFERENCES document_sections(section_id) ON DELETE CASCADE,
    FOREIGN KEY (value_id) REFERENCES core_values_taxonomy(value_id) ON DELETE CASCADE
);

-- Values Coding Table (Stage 1)
-- Human-validated values selections for each section
CREATE TABLE values_coding (
    coding_id INTEGER PRIMARY KEY,
    section_id INTEGER NOT NULL,            -- Reference to document_sections
    value_id INTEGER,                       -- Reference to core_values_taxonomy (NULL if custom)
    custom_value_name TEXT,                 -- For manual entries not in taxonomy
    is_manual_entry INTEGER DEFAULT 0,     -- 1 if manually entered, 0 if from taxonomy
    selected_from_suggestion INTEGER DEFAULT 0, -- 1 if selected from Claude suggestions
    confidence_level TEXT CHECK (confidence_level IN ('high', 'medium', 'low')),
    coder_notes TEXT,
    coded_date TEXT DEFAULT (datetime('now')),
    coder_name TEXT NOT NULL,
    locked_date TEXT,                       -- When the value selection was locked
    is_locked INTEGER DEFAULT 0,           -- 1 if locked (prevents modification)
    FOREIGN KEY (section_id) REFERENCES document_sections(section_id) ON DELETE CASCADE,
    FOREIGN KEY (value_id) REFERENCES core_values_taxonomy(value_id) ON DELETE SET NULL,
    UNIQUE(section_id, coder_name)          -- One values coding per section per coder
);

-- Document Sentences Table
-- Breaks coded sections into sentences for Stage 2 behavioral coding
CREATE TABLE document_sentences (
    sentence_id INTEGER PRIMARY KEY,
    section_id INTEGER NOT NULL,            -- Reference to document_sections
    sentence_number INTEGER NOT NULL,       -- Sequential sentence number within section
    sentence_text TEXT NOT NULL,            -- The actual sentence content
    start_pos INTEGER NOT NULL,             -- Character position where sentence starts (relative to section)
    end_pos INTEGER NOT NULL,               -- Character position where sentence ends (relative to section)
    created_date TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (section_id) REFERENCES document_sections(section_id) ON DELETE CASCADE,
    UNIQUE(section_id, sentence_number)
);

-- Claude Behavioral Suggestions Table
-- Stores AI-generated behavioral scale suggestions for sentences
CREATE TABLE claude_behavioral_suggestions (
    suggestion_id INTEGER PRIMARY KEY,
    sentence_id INTEGER NOT NULL,           -- Reference to document_sentences
    behavioral_score INTEGER NOT NULL CHECK (behavioral_score BETWEEN -3 AND 3), -- Scale: -3 to +3
    confidence_score REAL NOT NULL CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
    rationale TEXT,                         -- Claude's explanation for the behavioral score
    suggested_date TEXT DEFAULT (datetime('now')),
    model_version TEXT,                     -- Which Claude model version made the suggestion
    FOREIGN KEY (sentence_id) REFERENCES document_sentences(sentence_id) ON DELETE CASCADE
);

-- Behavioral Coding Table (Stage 2) 
-- Human-validated behavioral scale ratings for sentences
CREATE TABLE behavioral_coding (
    coding_id INTEGER PRIMARY KEY,
    sentence_id INTEGER NOT NULL,           -- Reference to document_sentences
    values_coding_id INTEGER NOT NULL,     -- Reference to the values coding this builds on
    behavioral_score INTEGER NOT NULL CHECK (behavioral_score BETWEEN -3 AND 3), -- Scale: -3 to +3
    selected_from_suggestion INTEGER DEFAULT 0, -- 1 if selected from Claude suggestions
    confidence_level TEXT CHECK (confidence_level IN ('high', 'medium', 'low')),
    coder_rationale TEXT,                   -- Human explanation for the rating
    coded_date TEXT DEFAULT (datetime('now')),
    coder_name TEXT NOT NULL,
    locked_date TEXT,                       -- When the behavioral rating was locked
    is_locked INTEGER DEFAULT 0,           -- 1 if locked (prevents modification)
    FOREIGN KEY (sentence_id) REFERENCES document_sentences(sentence_id) ON DELETE CASCADE,
    FOREIGN KEY (values_coding_id) REFERENCES values_coding(coding_id) ON DELETE CASCADE,
    UNIQUE(sentence_id, coder_name)         -- One behavioral coding per sentence per coder
);

-- Behavioral Scale Definitions Table
-- Stores the definitions for each point on the -3 to +3 behavioral scale
CREATE TABLE behavioral_scale_definitions (
    scale_point INTEGER PRIMARY KEY CHECK (scale_point BETWEEN -3 AND 3),
    scale_name TEXT NOT NULL,
    short_description TEXT NOT NULL,
    full_description TEXT NOT NULL,
    examples TEXT,
    created_date TEXT DEFAULT (datetime('now'))
);

-- Coding Progress Tracking Table
-- Tracks progress through the two-stage coding process
CREATE TABLE coding_progress (
    progress_id INTEGER PRIMARY KEY,
    session_id INTEGER NOT NULL,            -- Reference to values_coding_session
    fid INTEGER NOT NULL,                   -- Reference to source.id (file)
    total_sections INTEGER DEFAULT 0,       -- Total sections identified in document
    sections_values_coded INTEGER DEFAULT 0, -- Sections with completed values coding
    sections_values_locked INTEGER DEFAULT 0, -- Sections with locked values coding
    total_sentences INTEGER DEFAULT 0,      -- Total sentences in all coded sections
    sentences_behavioral_coded INTEGER DEFAULT 0, -- Sentences with completed behavioral coding
    sentences_behavioral_locked INTEGER DEFAULT 0, -- Sentences with locked behavioral coding
    stage_1_complete INTEGER DEFAULT 0,     -- 1 if all values coding is complete
    stage_2_complete INTEGER DEFAULT 0,     -- 1 if all behavioral coding is complete
    last_updated TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (session_id) REFERENCES values_coding_session(session_id) ON DELETE CASCADE,
    FOREIGN KEY (fid) REFERENCES source(id) ON DELETE CASCADE,
    UNIQUE(session_id, fid)
);

-- Export Templates Table
-- Stores templates for exporting values and behavioral analysis
CREATE TABLE export_templates (
    template_id INTEGER PRIMARY KEY,
    template_name TEXT NOT NULL UNIQUE,
    template_type TEXT NOT NULL CHECK (template_type IN ('values_summary', 'behavioral_summary', 'combined', 'detailed')),
    export_format TEXT NOT NULL CHECK (export_format IN ('csv', 'json', 'excel', 'pdf')),
    template_config TEXT,                   -- JSON configuration for the template
    created_date TEXT DEFAULT (datetime('now')),
    created_by TEXT,
    is_default INTEGER DEFAULT 0
);

-- Insert default core values from Phase 0 research
INSERT INTO core_values_taxonomy (value_name, value_category, description, definition) VALUES
-- Achievement Values
('Achievement', 'Achievement', 'Personal accomplishment and success', 'The drive to accomplish personal goals and demonstrate competence'),
('Excellence', 'Achievement', 'Pursuing the highest quality and standards', 'Commitment to superior performance and continuous improvement'),
('Recognition', 'Achievement', 'Seeking acknowledgment and appreciation', 'Desire for appreciation, respect, and acknowledgment from others'),
('Competence', 'Achievement', 'Developing and demonstrating skills', 'Building and showcasing professional and personal capabilities'),

-- Benevolence Values  
('Benevolence', 'Benevolence', 'Concern for the welfare of others', 'Acting with kindness and generosity toward others'),
('Service', 'Benevolence', 'Helping and supporting others', 'Dedication to meeting the needs of others and contributing to their wellbeing'),
('Compassion', 'Benevolence', 'Empathy and care for others', 'Deep concern for the suffering and needs of others'),
('Generosity', 'Benevolence', 'Giving freely to others', 'Willingness to share resources, time, and effort with others'),

-- Conformity Values
('Conformity', 'Conformity', 'Following rules and social expectations', 'Adherence to social norms, rules, and expectations'),
('Tradition', 'Conformity', 'Respecting customs and established practices', 'Commitment to maintaining cultural and organizational traditions'),
('Obedience', 'Conformity', 'Following authority and instructions', 'Compliance with legitimate authority and established procedures'),
('Politeness', 'Conformity', 'Social courtesy and respect', 'Maintaining proper social behavior and showing respect to others'),

-- Hedonism Values
('Hedonism', 'Hedonism', 'Seeking pleasure and enjoyment', 'Pursuit of personal pleasure, enjoyment, and gratification'),
('Fun', 'Hedonism', 'Engaging in enjoyable activities', 'Seeking entertainment, amusement, and lighthearted experiences'),
('Comfort', 'Hedonism', 'Seeking ease and convenience', 'Preference for comfortable, convenient, and stress-free conditions'),

-- Power Values
('Power', 'Power', 'Seeking control and influence', 'Desire for control, dominance, and influence over others and situations'),
('Authority', 'Power', 'Exercising leadership and control', 'Seeking positions of authority and decision-making responsibility'),
('Dominance', 'Power', 'Controlling others and situations', 'Desire to control and direct others behavior and outcomes'),
('Status', 'Power', 'Seeking high social position', 'Pursuit of prestige, social rank, and elevated social position'),

-- Security Values
('Security', 'Security', 'Seeking safety and stability', 'Desire for safety, stability, and predictability in life and work'),
('Safety', 'Security', 'Protecting from harm and danger', 'Ensuring protection from physical, emotional, and financial risks'),
('Stability', 'Security', 'Maintaining consistent conditions', 'Preference for predictable, steady, and unchanging circumstances'),
('Order', 'Security', 'Organization and systematic approach', 'Maintaining structure, organization, and systematic processes'),

-- Self-Direction Values
('Self-Direction', 'Self-Direction', 'Independence and autonomy', 'Acting independently and making autonomous decisions'),
('Independence', 'Self-Direction', 'Freedom from external control', 'Operating without external constraints or supervision'),
('Autonomy', 'Self-Direction', 'Self-governance and choice', 'Having the freedom to make ones own choices and decisions'),
('Creativity', 'Self-Direction', 'Innovation and original thinking', 'Developing new ideas, approaches, and creative solutions'),

-- Stimulation Values
('Stimulation', 'Stimulation', 'Seeking excitement and novelty', 'Pursuit of excitement, novelty, and challenging experiences'),
('Adventure', 'Stimulation', 'Seeking new and exciting experiences', 'Pursuing novel, thrilling, and adventurous activities'),
('Variety', 'Stimulation', 'Seeking diverse experiences', 'Preference for diverse, varied, and changing experiences'),
('Challenge', 'Stimulation', 'Embracing difficult tasks', 'Seeking difficult, demanding, and challenging work or situations'),

-- Universalism Values
('Universalism', 'Universalism', 'Concern for all people and nature', 'Care and concern for the welfare of all people and the natural world'),
('Justice', 'Universalism', 'Fairness and equality', 'Commitment to fairness, equality, and just treatment for all'),
('Equality', 'Universalism', 'Equal treatment and opportunities', 'Belief that all people deserve equal treatment and opportunities'),
('Environmental', 'Universalism', 'Protecting the natural world', 'Commitment to environmental protection and sustainability'),

-- Additional Core Values
('Integrity', 'Core', 'Honesty and moral consistency', 'Acting in accordance with moral and ethical principles consistently'),
('Respect', 'Core', 'Valuing others dignity and worth', 'Treating others with dignity, consideration, and appreciation'),
('Responsibility', 'Core', 'Accountability for actions and duties', 'Taking ownership of actions, decisions, and their consequences'),
('Trust', 'Core', 'Reliability and dependability', 'Being trustworthy and having confidence in others trustworthiness');

-- Insert behavioral scale definitions
INSERT INTO behavioral_scale_definitions (scale_point, scale_name, short_description, full_description, examples) VALUES
(-3, 'Extraordinary Violation', 'Systematic undermining of values', 'Deliberately and systematically acting in ways that undermine or contradict the core value, often with significant negative consequences', 'Deliberately sabotaging team efforts, systematically violating ethical standards, actively working against organizational values'),
(-2, 'Active Violation', 'Deliberate contradiction of values', 'Consciously and deliberately acting in ways that directly contradict or oppose the core value', 'Knowingly breaking rules for personal gain, deliberately treating others unfairly, actively resisting positive changes'),
(-1, 'Capitulating', 'Surrender through inaction', 'Failing to act in alignment with values when action was possible, essentially surrendering to opposing forces through passivity', 'Remaining silent when speaking up is needed, avoiding responsibility when leadership is required, giving up on important principles under pressure'),
(0, 'Indifference', 'Apathetic disengagement', 'Showing no particular care or concern for the value, neither supporting nor opposing it, remaining neutral or disengaged', 'Being indifferent to team success or failure, showing no concern for ethical issues, remaining uninvolved in important decisions'),
(1, 'Compromising', 'Partial selective enactment', 'Acting in alignment with the value only partially or selectively, making compromises that weaken full commitment', 'Following some rules but not others, helping some people but not others, pursuing quality only when convenient'),
(2, 'Active Enacting', 'Consistent deliberate alignment', 'Consistently and deliberately acting in ways that align with and support the core value, making it a priority in decisions and actions', 'Consistently following through on commitments, regularly helping team members, maintaining high standards even under pressure'),
(3, 'Extraordinary Enacting', 'Exceptional commitment with sacrifice', 'Going above and beyond to enact the value, often at significant personal or professional cost, demonstrating exceptional commitment', 'Sacrificing personal advancement to maintain integrity, taking significant risks to help others, persevering through extreme challenges to uphold principles');

-- Create indexes for performance optimization
CREATE INDEX idx_values_coding_session_fid ON values_coding_session(fid);
CREATE INDEX idx_values_coding_session_stage ON values_coding_session(stage);
CREATE INDEX idx_document_sections_fid ON document_sections(fid);
CREATE INDEX idx_document_sections_session ON document_sections(session_id);
CREATE INDEX idx_claude_values_suggestions_section ON claude_values_suggestions(section_id);
CREATE INDEX idx_claude_values_suggestions_value ON claude_values_suggestions(value_id);
CREATE INDEX idx_values_coding_section ON values_coding(section_id);
CREATE INDEX idx_values_coding_value ON values_coding(value_id);
CREATE INDEX idx_document_sentences_section ON document_sentences(section_id);
CREATE INDEX idx_claude_behavioral_suggestions_sentence ON claude_behavioral_suggestions(sentence_id);
CREATE INDEX idx_behavioral_coding_sentence ON behavioral_coding(sentence_id);
CREATE INDEX idx_behavioral_coding_values ON behavioral_coding(values_coding_id);
CREATE INDEX idx_coding_progress_session ON coding_progress(session_id);
CREATE INDEX idx_coding_progress_fid ON coding_progress(fid);