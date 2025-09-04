"""
Values Taxonomy MCP Server
Provides structured access to the 32-value taxonomy and custom values
"""

import json
import sqlite3
from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio
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

class ValuesTaxonomyServer:
    def __init__(self, database_path: str = None):
        self.database_path = database_path or "values_behavioral.db"
        self.server = Server("values-taxonomy") if Server else None
        
        # Initialize database if needed
        self._init_database()
        
        if self.server:
            self._register_tools()
    
    def _init_database(self):
        """Initialize database with taxonomy if not exists"""
        conn = sqlite3.connect(self.database_path)
        try:
            # Check if taxonomy table exists
            cursor = conn.cursor()
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='core_values_taxonomy'
            """)
            
            if not cursor.fetchone():
                # Create and populate taxonomy table
                self._create_taxonomy_schema(conn)
                self._populate_default_taxonomy(conn)
        finally:
            conn.close()
    
    def _create_taxonomy_schema(self, conn: sqlite3.Connection):
        """Create the taxonomy schema"""
        conn.executescript("""
            CREATE TABLE core_values_taxonomy (
                value_id INTEGER PRIMARY KEY,
                value_name TEXT NOT NULL UNIQUE,
                value_category TEXT NOT NULL,
                description TEXT,
                definition TEXT,
                created_date TEXT DEFAULT (datetime('now')),
                is_active INTEGER DEFAULT 1
            );
            
            CREATE TABLE custom_values (
                custom_id INTEGER PRIMARY KEY,
                value_name TEXT NOT NULL,
                category TEXT,
                description TEXT,
                created_by TEXT,
                created_date TEXT DEFAULT (datetime('now')),
                usage_count INTEGER DEFAULT 0
            );
            
            CREATE TABLE value_statistics (
                value_id INTEGER,
                custom_id INTEGER,
                document_count INTEGER DEFAULT 0,
                section_count INTEGER DEFAULT 0,
                confidence_avg REAL DEFAULT 0.0,
                last_updated TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (value_id) REFERENCES core_values_taxonomy(value_id),
                FOREIGN KEY (custom_id) REFERENCES custom_values(custom_id)
            );
        """)
    
    def _populate_default_taxonomy(self, conn: sqlite3.Connection):
        """Populate with the 32 core values"""
        values = [
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
        ]
        
        conn.executemany(
            "INSERT INTO core_values_taxonomy (value_name, value_category, description, definition) VALUES (?, ?, ?, ?)",
            values
        )
        conn.commit()
    
    def _register_tools(self):
        """Register MCP tools"""
        @self.server.call_tool()
        async def get_taxonomy_values(category: Optional[str] = None) -> List[TextContent]:
            """Get all values from taxonomy, optionally filtered by category"""
            conn = sqlite3.connect(self.database_path)
            try:
                cursor = conn.cursor()
                if category:
                    cursor.execute("""
                        SELECT value_id, value_name, value_category, description, definition
                        FROM core_values_taxonomy 
                        WHERE value_category = ? AND is_active = 1
                        ORDER BY value_name
                    """, (category,))
                else:
                    cursor.execute("""
                        SELECT value_id, value_name, value_category, description, definition
                        FROM core_values_taxonomy 
                        WHERE is_active = 1
                        ORDER BY value_category, value_name
                    """)
                
                values = []
                for row in cursor.fetchall():
                    values.append({
                        'value_id': row[0],
                        'value_name': row[1],
                        'value_category': row[2],
                        'description': row[3],
                        'definition': row[4]
                    })
                
                return [TextContent(
                    type="text",
                    text=json.dumps({"values": values}, indent=2)
                )]
            finally:
                conn.close()
        
        @self.server.call_tool()
        async def search_values_by_category(category: str) -> List[TextContent]:
            """Search for values by category"""
            return await get_taxonomy_values(category=category)
        
        @self.server.call_tool()
        async def add_custom_value(value_name: str, category: str, description: str, created_by: str) -> List[TextContent]:
            """Add a new custom value to the system"""
            conn = sqlite3.connect(self.database_path)
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO custom_values (value_name, category, description, created_by)
                    VALUES (?, ?, ?, ?)
                """, (value_name, category, description, created_by))
                
                custom_id = cursor.lastrowid
                conn.commit()
                
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "success": True,
                        "custom_id": custom_id,
                        "message": f"Custom value '{value_name}' added successfully"
                    })
                )]
            except sqlite3.IntegrityError as e:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "success": False,
                        "error": f"Value may already exist: {str(e)}"
                    })
                )]
            finally:
                conn.close()
        
        @self.server.call_tool()
        async def update_value_definition(value_id: int, new_definition: str) -> List[TextContent]:
            """Update the definition of an existing taxonomy value"""
            conn = sqlite3.Connection(self.database_path)
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE core_values_taxonomy 
                    SET definition = ?, created_date = datetime('now')
                    WHERE value_id = ?
                """, (new_definition, value_id))
                
                if cursor.rowcount > 0:
                    conn.commit()
                    return [TextContent(
                        type="text",
                        text=json.dumps({
                            "success": True,
                            "message": f"Value {value_id} definition updated"
                        })
                    )]
                else:
                    return [TextContent(
                        type="text",
                        text=json.dumps({
                            "success": False,
                            "error": f"Value {value_id} not found"
                        })
                    )]
            finally:
                conn.close()
        
        @self.server.call_tool()
        async def get_value_statistics(value_id: Optional[int] = None) -> List[TextContent]:
            """Get usage statistics for values"""
            conn = sqlite3.connect(self.database_path)
            try:
                cursor = conn.cursor()
                if value_id:
                    cursor.execute("""
                        SELECT v.value_name, v.value_category, 
                               COALESCE(s.document_count, 0) as doc_count,
                               COALESCE(s.section_count, 0) as section_count,
                               COALESCE(s.confidence_avg, 0.0) as avg_confidence
                        FROM core_values_taxonomy v
                        LEFT JOIN value_statistics s ON v.value_id = s.value_id
                        WHERE v.value_id = ?
                    """, (value_id,))
                    
                    row = cursor.fetchone()
                    if row:
                        stats = {
                            'value_name': row[0],
                            'value_category': row[1],
                            'document_count': row[2],
                            'section_count': row[3],
                            'average_confidence': row[4]
                        }
                    else:
                        stats = {"error": "Value not found"}
                else:
                    cursor.execute("""
                        SELECT v.value_name, v.value_category,
                               COALESCE(s.document_count, 0) as doc_count,
                               COALESCE(s.section_count, 0) as section_count,
                               COALESCE(s.confidence_avg, 0.0) as avg_confidence
                        FROM core_values_taxonomy v
                        LEFT JOIN value_statistics s ON v.value_id = s.value_id
                        WHERE v.is_active = 1
                        ORDER BY s.section_count DESC, v.value_name
                    """)
                    
                    stats = []
                    for row in cursor.fetchall():
                        stats.append({
                            'value_name': row[0],
                            'value_category': row[1],
                            'document_count': row[2],
                            'section_count': row[3],
                            'average_confidence': row[4]
                        })
                
                return [TextContent(
                    type="text",
                    text=json.dumps({"statistics": stats}, indent=2)
                )]
            finally:
                conn.close()

async def main():
    """Run the Values Taxonomy MCP Server"""
    if not Server:
        print("MCP library not available. Install with: pip install mcp")
        return
    
    server = ValuesTaxonomyServer()
    
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.server.run(
            read_stream,
            write_stream,
            server.server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())