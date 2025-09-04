"""
Document Processing MCP Server
Handles text segmentation and document parsing for values coding
"""

import json
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import sqlite3
import spacy
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

class DocumentSegment:
    def __init__(self, text: str, start_pos: int, end_pos: int, segment_type: str = "paragraph"):
        self.text = text.strip()
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.segment_type = segment_type
        self.word_count = len(self.text.split())
        self.sentence_count = len(self.get_sentences())
    
    def get_sentences(self) -> List[str]:
        """Extract sentences from the segment"""
        # Simple sentence splitting - could be enhanced with spaCy
        sentences = re.split(r'[.!?]+', self.text)
        return [s.strip() for s in sentences if s.strip()]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'text': self.text,
            'start_pos': self.start_pos,
            'end_pos': self.end_pos,
            'segment_type': self.segment_type,
            'word_count': self.word_count,
            'sentence_count': self.sentence_count
        }

class DocumentProcessingServer:
    def __init__(self, database_path: str = None):
        self.database_path = database_path or "values_behavioral.db"
        self.server = Server("document-processing") if Server else None
        self.nlp = None
        
        # Try to load spaCy model
        self._init_nlp()
        
        if self.server:
            self._register_tools()
    
    def _init_nlp(self):
        """Initialize spaCy NLP pipeline"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("spaCy model 'en_core_web_sm' not found. Using basic text processing.")
            print("Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def _register_tools(self):
        """Register MCP tools"""
        
        @self.server.call_tool()
        async def segment_document(text: str, segment_type: str = "paragraph", min_words: int = 20, max_words: int = 150) -> List[TextContent]:
            """Segment document into analyzable chunks"""
            segments = []
            
            if segment_type == "paragraph":
                segments = self._segment_by_paragraphs(text, min_words, max_words)
            elif segment_type == "sentence":
                segments = self._segment_by_sentences(text, min_words, max_words)
            elif segment_type == "smart":
                segments = self._smart_segmentation(text, min_words, max_words)
            else:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": f"Unknown segment_type: {segment_type}"})
                )]
            
            result = {
                "segments": [seg.to_dict() for seg in segments],
                "total_segments": len(segments),
                "total_words": sum(seg.word_count for seg in segments),
                "segmentation_type": segment_type
            }
            
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
        
        @self.server.call_tool()
        async def parse_sentences(text: str, segment_start: int = 0) -> List[TextContent]:
            """Parse text into individual sentences with positions"""
            sentences = []
            
            if self.nlp:
                # Use spaCy for better sentence detection
                doc = self.nlp(text)
                for sent in doc.sents:
                    sentences.append({
                        'text': sent.text.strip(),
                        'start_pos': segment_start + sent.start_char,
                        'end_pos': segment_start + sent.end_char,
                        'word_count': len(sent.text.split())
                    })
            else:
                # Basic sentence splitting
                import re
                sentence_pattern = r'[.!?]+\s+'
                parts = re.split(sentence_pattern, text)
                current_pos = segment_start
                
                for part in parts:
                    if part.strip():
                        sentences.append({
                            'text': part.strip(),
                            'start_pos': current_pos,
                            'end_pos': current_pos + len(part),
                            'word_count': len(part.split())
                        })
                        current_pos += len(part) + 2  # Account for delimiter
            
            return [TextContent(
                type="text",
                text=json.dumps({"sentences": sentences}, indent=2)
            )]
        
        @self.server.call_tool()
        async def extract_contexts(text: str, target_phrases: List[str], context_window: int = 50) -> List[TextContent]:
            """Extract context around specific phrases or terms"""
            contexts = []
            
            for phrase in target_phrases:
                pattern = re.compile(re.escape(phrase), re.IGNORECASE)
                for match in pattern.finditer(text):
                    start = max(0, match.start() - context_window)
                    end = min(len(text), match.end() + context_window)
                    
                    contexts.append({
                        'phrase': phrase,
                        'context': text[start:end],
                        'phrase_start': match.start(),
                        'phrase_end': match.end(),
                        'context_start': start,
                        'context_end': end
                    })
            
            return [TextContent(
                type="text",
                text=json.dumps({"contexts": contexts}, indent=2)
            )]
        
        @self.server.call_tool()
        async def validate_segments(segments_json: str, min_words: int = 10, max_words: int = 200) -> List[TextContent]:
            """Validate that segments meet quality criteria"""
            try:
                segments_data = json.loads(segments_json)
                segments = segments_data.get("segments", [])
            except json.JSONDecodeError:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": "Invalid JSON format"})
                )]
            
            validation_results = []
            
            for i, segment in enumerate(segments):
                issues = []
                word_count = segment.get('word_count', 0)
                text = segment.get('text', '')
                
                # Check word count bounds
                if word_count < min_words:
                    issues.append(f"Too short ({word_count} words, minimum {min_words})")
                if word_count > max_words:
                    issues.append(f"Too long ({word_count} words, maximum {max_words})")
                
                # Check for empty or whitespace-only text
                if not text.strip():
                    issues.append("Empty or whitespace-only text")
                
                # Check for meaningful content
                if len(text.split()) < 3:
                    issues.append("Insufficient meaningful content")
                
                # Check sentence structure
                if not re.search(r'[.!?]', text):
                    issues.append("No sentence-ending punctuation")
                
                validation_results.append({
                    'segment_index': i,
                    'is_valid': len(issues) == 0,
                    'issues': issues,
                    'word_count': word_count,
                    'text_preview': text[:100] + '...' if len(text) > 100 else text
                })
            
            summary = {
                'total_segments': len(segments),
                'valid_segments': sum(1 for r in validation_results if r['is_valid']),
                'invalid_segments': sum(1 for r in validation_results if not r['is_valid']),
                'validation_results': validation_results
            }
            
            return [TextContent(
                type="text",
                text=json.dumps(summary, indent=2)
            )]
        
        @self.server.call_tool()
        async def get_document_metadata(text: str) -> List[TextContent]:
            """Extract metadata and statistics from document"""
            # Basic text statistics
            word_count = len(text.split())
            char_count = len(text)
            char_count_no_spaces = len(text.replace(' ', ''))
            
            # Paragraph and sentence counts
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            paragraph_count = len(paragraphs)
            
            # Sentence count (approximate)
            sentence_count = len(re.findall(r'[.!?]+', text))
            
            # Average sentence length
            avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
            
            # Reading complexity (simple measure)
            avg_word_length = sum(len(word) for word in text.split()) / word_count if word_count > 0 else 0
            
            # Language detection and analysis with spaCy if available
            language_features = {}
            if self.nlp:
                doc = self.nlp(text[:1000])  # Analyze first 1000 chars for efficiency
                pos_tags = [token.pos_ for token in doc]
                language_features = {
                    'noun_phrases': len([chunk for chunk in doc.noun_chunks]),
                    'named_entities': len([ent for ent in doc.ents]),
                    'most_common_pos': max(set(pos_tags), key=pos_tags.count) if pos_tags else None
                }
            
            metadata = {
                'word_count': word_count,
                'character_count': char_count,
                'character_count_no_spaces': char_count_no_spaces,
                'paragraph_count': paragraph_count,
                'sentence_count': sentence_count,
                'average_sentence_length': round(avg_sentence_length, 2),
                'average_word_length': round(avg_word_length, 2),
                'language_features': language_features,
                'suggested_segment_count': max(1, word_count // 75),  # ~75 words per segment
                'processing_timestamp': datetime.now().isoformat()
            }
            
            return [TextContent(
                type="text",
                text=json.dumps(metadata, indent=2)
            )]
    
    def _segment_by_paragraphs(self, text: str, min_words: int, max_words: int) -> List[DocumentSegment]:
        """Segment text by paragraphs"""
        segments = []
        paragraphs = text.split('\n\n')
        current_pos = 0
        
        for para in paragraphs:
            para = para.strip()
            if para:
                word_count = len(para.split())
                
                if word_count >= min_words:
                    if word_count <= max_words:
                        # Perfect size paragraph
                        segments.append(DocumentSegment(
                            text=para,
                            start_pos=current_pos,
                            end_pos=current_pos + len(para),
                            segment_type="paragraph"
                        ))
                    else:
                        # Split large paragraph
                        sub_segments = self._split_large_text(para, current_pos, max_words)
                        segments.extend(sub_segments)
                
            current_pos += len(para) + 2  # Account for \n\n
        
        return segments
    
    def _segment_by_sentences(self, text: str, min_words: int, max_words: int) -> List[DocumentSegment]:
        """Segment text by combining sentences to meet word count requirements"""
        if self.nlp:
            doc = self.nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents]
        else:
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
        
        segments = []
        current_segment = []
        current_words = 0
        current_pos = 0
        
        for sentence in sentences:
            sentence_words = len(sentence.split())
            
            if current_words + sentence_words <= max_words:
                current_segment.append(sentence)
                current_words += sentence_words
            else:
                # Finalize current segment if it meets minimum
                if current_words >= min_words:
                    segment_text = '. '.join(current_segment) + '.'
                    segments.append(DocumentSegment(
                        text=segment_text,
                        start_pos=current_pos,
                        end_pos=current_pos + len(segment_text),
                        segment_type="sentence_group"
                    ))
                    current_pos += len(segment_text) + 1
                
                # Start new segment
                current_segment = [sentence]
                current_words = sentence_words
        
        # Handle final segment
        if current_segment and current_words >= min_words:
            segment_text = '. '.join(current_segment) + '.'
            segments.append(DocumentSegment(
                text=segment_text,
                start_pos=current_pos,
                end_pos=current_pos + len(segment_text),
                segment_type="sentence_group"
            ))
        
        return segments
    
    def _smart_segmentation(self, text: str, min_words: int, max_words: int) -> List[DocumentSegment]:
        """Intelligent segmentation considering context and meaning"""
        # Start with paragraph segmentation, then refine
        segments = self._segment_by_paragraphs(text, min_words, max_words)
        
        # Merge very short adjacent segments
        merged_segments = []
        i = 0
        while i < len(segments):
            current_segment = segments[i]
            
            # Look ahead to merge short segments
            if (current_segment.word_count < min_words and 
                i + 1 < len(segments) and
                current_segment.word_count + segments[i + 1].word_count <= max_words):
                
                # Merge with next segment
                next_segment = segments[i + 1]
                merged_text = current_segment.text + '\n\n' + next_segment.text
                merged_segments.append(DocumentSegment(
                    text=merged_text,
                    start_pos=current_segment.start_pos,
                    end_pos=next_segment.end_pos,
                    segment_type="merged"
                ))
                i += 2  # Skip next segment
            else:
                merged_segments.append(current_segment)
                i += 1
        
        return merged_segments
    
    def _split_large_text(self, text: str, start_pos: int, max_words: int) -> List[DocumentSegment]:
        """Split text that's too large into smaller segments"""
        segments = []
        words = text.split()
        current_pos = start_pos
        
        for i in range(0, len(words), max_words):
            chunk_words = words[i:i + max_words]
            chunk_text = ' '.join(chunk_words)
            
            segments.append(DocumentSegment(
                text=chunk_text,
                start_pos=current_pos,
                end_pos=current_pos + len(chunk_text),
                segment_type="split_chunk"
            ))
            current_pos += len(chunk_text) + 1
        
        return segments

async def main():
    """Run the Document Processing MCP Server"""
    if not Server:
        print("MCP library not available. Install with: pip install mcp")
        return
    
    server = DocumentProcessingServer()
    
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.server.run(
            read_stream,
            write_stream,
            server.server.create_initialization_options()
        )

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())