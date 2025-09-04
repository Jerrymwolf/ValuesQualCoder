-- Open Coding Extension Schema
-- Additional tables to support open coding and taxonomy validation

-- Open Coded Values Table
-- Stores values identified through unrestricted open coding
CREATE TABLE open_coded_values (
    open_value_id INTEGER PRIMARY KEY,
    section_id INTEGER NOT NULL,
    value_name TEXT NOT NULL,
    suggested_category TEXT,
    confidence_score REAL NOT NULL CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
    rationale TEXT,
    coded_date TEXT DEFAULT (datetime('now')),
    coder_name TEXT NOT NULL,
    model_version TEXT,
    is_validated INTEGER DEFAULT 0,
    FOREIGN KEY (section_id) REFERENCES document_sections(section_id) ON DELETE CASCADE
);

-- Taxonomy Mappings Table
-- Maps open-coded values to taxonomy values
CREATE TABLE taxonomy_mappings (
    mapping_id INTEGER PRIMARY KEY,
    open_value_id INTEGER NOT NULL,
    taxonomy_value_id INTEGER,
    mapping_type TEXT NOT NULL CHECK (mapping_type IN ('direct_match', 'broader_category', 'taxonomy_gap', 'too_specific', 'no_match')),
    confidence_score REAL NOT NULL CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
    rationale TEXT,
    created_date TEXT DEFAULT (datetime('now')),
    created_by TEXT NOT NULL,
    validated_by TEXT,
    validation_date TEXT,
    FOREIGN KEY (open_value_id) REFERENCES open_coded_values(open_value_id) ON DELETE CASCADE,
    FOREIGN KEY (taxonomy_value_id) REFERENCES core_values_taxonomy(value_id) ON DELETE SET NULL
);

-- Taxonomy Recommendations Table
-- Stores recommendations for improving the taxonomy based on open coding
CREATE TABLE taxonomy_recommendations (
    recommendation_id INTEGER PRIMARY KEY,
    recommendation_type TEXT NOT NULL CHECK (recommendation_type IN ('add_value', 'modify_value', 'merge_values', 'split_value', 'recategorize')),
    current_value_name TEXT,
    suggested_value_name TEXT,
    suggested_category TEXT,
    rationale TEXT NOT NULL,
    supporting_evidence TEXT,
    frequency_count INTEGER DEFAULT 1,
    created_date TEXT DEFAULT (datetime('now')),
    created_by TEXT NOT NULL,
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'rejected', 'implemented'))
);

-- Taxonomy Validation Sessions Table
-- Tracks validation sessions comparing open coding to taxonomy
CREATE TABLE taxonomy_validation_sessions (
    validation_session_id INTEGER PRIMARY KEY,
    session_name TEXT NOT NULL,
    description TEXT,
    total_sections INTEGER DEFAULT 0,
    processed_sections INTEGER DEFAULT 0,
    identified_gaps INTEGER DEFAULT 0,
    mapping_accuracy REAL DEFAULT 0.0,
    created_date TEXT DEFAULT (datetime('now')),
    created_by TEXT NOT NULL,
    status TEXT DEFAULT 'active' CHECK (status IN ('active', 'completed', 'paused'))
);

-- Link table for validation sessions and open coding sessions
CREATE TABLE validation_open_coding_link (
    link_id INTEGER PRIMARY KEY,
    validation_session_id INTEGER NOT NULL,
    coding_session_id INTEGER NOT NULL,
    FOREIGN KEY (validation_session_id) REFERENCES taxonomy_validation_sessions(validation_session_id) ON DELETE CASCADE,
    FOREIGN KEY (coding_session_id) REFERENCES values_coding_session(session_id) ON DELETE CASCADE,
    UNIQUE(validation_session_id, coding_session_id)
);

-- Hybrid Coding Results Table
-- Stores results when using both open and taxonomy coding simultaneously
CREATE TABLE hybrid_coding_results (
    hybrid_result_id INTEGER PRIMARY KEY,
    section_id INTEGER NOT NULL,
    open_values_json TEXT, -- JSON array of open values
    taxonomy_matches_json TEXT, -- JSON array of taxonomy matches
    taxonomy_gaps_json TEXT, -- JSON array of identified gaps
    confidence_score REAL NOT NULL CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
    created_date TEXT DEFAULT (datetime('now')),
    created_by TEXT NOT NULL,
    FOREIGN KEY (section_id) REFERENCES document_sections(section_id) ON DELETE CASCADE
);

-- Update values_coding_session table to support coding modes
-- Add coding_mode column to track the type of coding being performed
ALTER TABLE values_coding_session ADD COLUMN coding_mode TEXT DEFAULT 'taxonomy' 
    CHECK (coding_mode IN ('open', 'taxonomy', 'hybrid', 'validation'));

-- Update coding_progress table to track open coding progress
ALTER TABLE coding_progress ADD COLUMN open_values_identified INTEGER DEFAULT 0;
ALTER TABLE coding_progress ADD COLUMN values_mapped INTEGER DEFAULT 0;
ALTER TABLE coding_progress ADD COLUMN taxonomy_gaps_found INTEGER DEFAULT 0;

-- Create indexes for performance
CREATE INDEX idx_open_coded_values_section ON open_coded_values(section_id);
CREATE INDEX idx_open_coded_values_name ON open_coded_values(value_name);
CREATE INDEX idx_open_coded_values_category ON open_coded_values(suggested_category);
CREATE INDEX idx_taxonomy_mappings_open_value ON taxonomy_mappings(open_value_id);
CREATE INDEX idx_taxonomy_mappings_taxonomy_value ON taxonomy_mappings(taxonomy_value_id);
CREATE INDEX idx_taxonomy_mappings_type ON taxonomy_mappings(mapping_type);
CREATE INDEX idx_taxonomy_recommendations_type ON taxonomy_recommendations(recommendation_type);
CREATE INDEX idx_taxonomy_recommendations_status ON taxonomy_recommendations(status);
CREATE INDEX idx_hybrid_coding_section ON hybrid_coding_results(section_id);
CREATE INDEX idx_validation_sessions_status ON taxonomy_validation_sessions(status);

-- Create views for common queries

-- View for open coding analysis
CREATE VIEW open_coding_analysis AS
SELECT 
    ocv.open_value_id,
    ocv.section_id,
    ocv.value_name,
    ocv.suggested_category,
    ocv.confidence_score,
    ocv.rationale,
    ocv.coder_name,
    ds.section_text,
    ds.fid,
    s.name as file_name,
    tm.mapping_type,
    tm.taxonomy_value_id,
    cvt.value_name as taxonomy_match
FROM open_coded_values ocv
JOIN document_sections ds ON ocv.section_id = ds.section_id
JOIN source s ON ds.fid = s.id
LEFT JOIN taxonomy_mappings tm ON ocv.open_value_id = tm.open_value_id
LEFT JOIN core_values_taxonomy cvt ON tm.taxonomy_value_id = cvt.value_id;

-- View for taxonomy gaps analysis
CREATE VIEW taxonomy_gaps_analysis AS
SELECT 
    ocv.value_name as open_value,
    ocv.suggested_category,
    COUNT(*) as frequency,
    AVG(ocv.confidence_score) as avg_confidence,
    GROUP_CONCAT(DISTINCT ocv.coder_name) as identified_by,
    tm.mapping_type
FROM open_coded_values ocv
LEFT JOIN taxonomy_mappings tm ON ocv.open_value_id = tm.open_value_id
WHERE tm.mapping_type = 'taxonomy_gap' OR tm.mapping_type IS NULL
GROUP BY ocv.value_name, ocv.suggested_category
ORDER BY frequency DESC, avg_confidence DESC;

-- View for mapping success rates
CREATE VIEW mapping_success_rates AS
SELECT 
    tm.mapping_type,
    COUNT(*) as count,
    AVG(tm.confidence_score) as avg_confidence,
    COUNT(*) * 100.0 / (SELECT COUNT(*) FROM taxonomy_mappings) as percentage
FROM taxonomy_mappings tm
GROUP BY tm.mapping_type
ORDER BY count DESC;