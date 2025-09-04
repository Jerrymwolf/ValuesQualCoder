"""
Values and Behavioral Enactment Coder - AI Integration
Claude AI integration for values identification and behavioral coding suggestions
"""

import json
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from .values_behavioral_constants import CLAUDE_PROMPTS, CORE_VALUES_TAXONOMY, BEHAVIORAL_SCALE
from .values_behavioral_models import (
    ClaudeValuesSuggestion, ClaudeBehavioralSuggestion, 
    ValuesBehavioralException, ClaudeAPIError
)

logger = logging.getLogger(__name__)


class ValuesBehavioralAI:
    """AI service for values and behavioral coding using Claude"""
    
    def __init__(self, app):
        """Initialize with app instance"""
        self.app = app
        self.ai_service = app.ai if hasattr(app, 'ai') else None
        self.model_version = "claude-3-5-sonnet-20241022"  # Default Claude model
        
        # Cache for repeated calls
        self._values_cache: Dict[str, List[Dict]] = {}
        self._behavioral_cache: Dict[Tuple[str, str], Dict] = {}
        
    def is_available(self) -> bool:
        """Check if AI service is available"""
        return self.ai_service is not None and hasattr(self.ai_service, 'is_ready') and self.ai_service.is_ready()
    
    async def get_values_suggestions(self, text_section: str, use_cache: bool = True) -> List[Dict[str, Any]]:
        """Get Claude suggestions for values identification
        
        Args:
            text_section: The text section to analyze
            use_cache: Whether to use cached results
            
        Returns:
            List of value suggestions with confidence scores and rationales
        """
        try:
            # Check cache first
            if use_cache and text_section in self._values_cache:
                logger.info("Using cached values suggestions")
                return self._values_cache[text_section]
            
            if not self.is_available():
                raise ClaudeAPIError("AI service not available")
            
            # Prepare values list for prompt
            values_list = self._format_values_list()
            
            # Create prompt
            prompt = CLAUDE_PROMPTS["values_identification"].format(
                values_list=values_list,
                text_section=text_section
            )
            
            # Get AI response
            response = await self._get_ai_completion(prompt)
            
            # Parse JSON response
            try:
                result = json.loads(response)
                suggestions = result.get('values', [])
            except json.JSONDecodeError:
                # Try to repair malformed JSON
                import json_repair
                repaired = json_repair.repair_json(response)
                result = json.loads(repaired)
                suggestions = result.get('values', [])
            
            # Validate and process suggestions
            processed_suggestions = self._process_values_suggestions(suggestions)
            
            # Cache results
            if use_cache:
                self._values_cache[text_section] = processed_suggestions
            
            logger.info(f"Generated {len(processed_suggestions)} values suggestions")
            return processed_suggestions
            
        except Exception as e:
            logger.error(f"Error getting values suggestions: {e}")
            raise ClaudeAPIError(f"Failed to get values suggestions: {e}")
    
    async def get_behavioral_suggestion(self, sentence_text: str, selected_value: str, 
                                      use_cache: bool = True) -> Dict[str, Any]:
        """Get Claude suggestion for behavioral coding
        
        Args:
            sentence_text: The sentence to analyze
            selected_value: The selected value for context
            use_cache: Whether to use cached results
            
        Returns:
            Dictionary with behavioral score, confidence, and rationale
        """
        try:
            cache_key = (sentence_text, selected_value)
            
            # Check cache first
            if use_cache and cache_key in self._behavioral_cache:
                logger.info("Using cached behavioral suggestion")
                return self._behavioral_cache[cache_key]
            
            if not self.is_available():
                raise ClaudeAPIError("AI service not available")
            
            # Create prompt
            prompt = CLAUDE_PROMPTS["behavioral_coding"].format(
                selected_value=selected_value,
                sentence_text=sentence_text
            )
            
            # Get AI response
            response = await self._get_ai_completion(prompt)
            
            # Parse JSON response
            try:
                result = json.loads(response)
            except json.JSONDecodeError:
                # Try to repair malformed JSON
                import json_repair
                repaired = json_repair.repair_json(response)
                result = json.loads(repaired)
            
            # Validate and process suggestion
            processed_suggestion = self._process_behavioral_suggestion(result)
            
            # Cache results
            if use_cache:
                self._behavioral_cache[cache_key] = processed_suggestion
            
            logger.info(f"Generated behavioral suggestion: {processed_suggestion['behavioral_score']}")
            return processed_suggestion
            
        except Exception as e:
            logger.error(f"Error getting behavioral suggestion: {e}")
            raise ClaudeAPIError(f"Failed to get behavioral suggestion: {e}")
    
    async def parse_text_sections(self, full_text: str) -> List[Dict[str, Any]]:
        """Use AI to parse text into logical sections
        
        Args:
            full_text: The complete text to parse
            
        Returns:
            List of section dictionaries with text, start_pos, end_pos
        """
        try:
            if not self.is_available():
                raise ClaudeAPIError("AI service not available")
            
            # Create prompt
            prompt = CLAUDE_PROMPTS["section_parsing"].format(full_text=full_text)
            
            # Get AI response
            response = await self._get_ai_completion(prompt)
            
            # Parse JSON response
            try:
                result = json.loads(response)
                sections = result.get('sections', [])
            except json.JSONDecodeError:
                # Try to repair malformed JSON
                import json_repair
                repaired = json_repair.repair_json(response)
                result = json.loads(repaired)
                sections = result.get('sections', [])
            
            # Validate sections
            processed_sections = []
            for section in sections:
                if all(key in section for key in ['section_text', 'start_pos', 'end_pos']):
                    processed_sections.append({
                        'text': section['section_text'],
                        'start_pos': section['start_pos'],
                        'end_pos': section['end_pos'],
                        'section_type': section.get('section_type', 'paragraph')
                    })
            
            logger.info(f"Parsed {len(processed_sections)} sections using AI")
            return processed_sections
            
        except Exception as e:
            logger.error(f"Error parsing text sections: {e}")
            raise ClaudeAPIError(f"Failed to parse text sections: {e}")
    
    async def parse_section_sentences(self, section_text: str) -> List[Dict[str, Any]]:
        """Use AI to parse section into sentences
        
        Args:
            section_text: The section text to parse
            
        Returns:
            List of sentence dictionaries with text, start_pos, end_pos
        """
        try:
            if not self.is_available():
                raise ClaudeAPIError("AI service not available")
            
            # Create prompt
            prompt = CLAUDE_PROMPTS["sentence_parsing"].format(section_text=section_text)
            
            # Get AI response
            response = await self._get_ai_completion(prompt)
            
            # Parse JSON response
            try:
                result = json.loads(response)
                sentences = result.get('sentences', [])
            except json.JSONDecodeError:
                # Try to repair malformed JSON
                import json_repair
                repaired = json_repair.repair_json(response)
                result = json.loads(repaired)
                sentences = result.get('sentences', [])
            
            # Validate sentences
            processed_sentences = []
            for sentence in sentences:
                if all(key in sentence for key in ['sentence_text', 'start_pos', 'end_pos']):
                    processed_sentences.append({
                        'text': sentence['sentence_text'],
                        'start_pos': sentence['start_pos'],
                        'end_pos': sentence['end_pos']
                    })
            
            logger.info(f"Parsed {len(processed_sentences)} sentences using AI")
            return processed_sentences
            
        except Exception as e:
            logger.error(f"Error parsing section sentences: {e}")
            raise ClaudeAPIError(f"Failed to parse section sentences: {e}")
    
    def get_values_suggestions_sync(self, text_section: str, use_cache: bool = True) -> List[Dict[str, Any]]:
        """Synchronous wrapper for get_values_suggestions"""
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.get_values_suggestions(text_section, use_cache))
        except RuntimeError:
            # Create new event loop if none exists
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.get_values_suggestions(text_section, use_cache))
            finally:
                loop.close()
    
    def get_behavioral_suggestion_sync(self, sentence_text: str, selected_value: str, 
                                     use_cache: bool = True) -> Dict[str, Any]:
        """Synchronous wrapper for get_behavioral_suggestion"""
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.get_behavioral_suggestion(sentence_text, selected_value, use_cache))
        except RuntimeError:
            # Create new event loop if none exists
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.get_behavioral_suggestion(sentence_text, selected_value, use_cache))
            finally:
                loop.close()
    
    # Private helper methods
    
    async def _get_ai_completion(self, prompt: str) -> str:
        """Get completion from AI service"""
        try:
            if hasattr(self.ai_service, 'get_completion_async'):
                return await self.ai_service.get_completion_async(prompt)
            elif hasattr(self.ai_service, 'get_completion'):
                # Fallback to synchronous method
                return self.ai_service.get_completion(prompt)
            else:
                raise ClaudeAPIError("No completion method available in AI service")
        except Exception as e:
            raise ClaudeAPIError(f"AI completion failed: {e}")
    
    def _format_values_list(self) -> str:
        """Format the core values taxonomy for the prompt"""
        values_text = []
        for category, values in CORE_VALUES_TAXONOMY.items():
            values_text.append(f"\n## {category}")
            for value_name, description, definition in values:
                values_text.append(f"- **{value_name}**: {description}")
        
        return "\n".join(values_text)
    
    def _process_values_suggestions(self, suggestions: List[Dict]) -> List[Dict[str, Any]]:
        """Process and validate values suggestions"""
        processed = []
        valid_values = set()
        
        # Get all valid value names
        for category, values in CORE_VALUES_TAXONOMY.items():
            for value_name, _, _ in values:
                valid_values.add(value_name)
        
        for suggestion in suggestions:
            try:
                value_name = suggestion.get('value', '').strip()
                confidence = float(suggestion.get('confidence', 0.0))
                rationale = suggestion.get('rationale', '').strip()
                
                # Validate value name
                if value_name not in valid_values:
                    logger.warning(f"Invalid value name in suggestion: {value_name}")
                    continue
                
                # Validate confidence
                if not (0.0 <= confidence <= 1.0):
                    logger.warning(f"Invalid confidence score: {confidence}")
                    confidence = max(0.0, min(1.0, confidence))
                
                processed.append({
                    'value': value_name,
                    'confidence': confidence,
                    'rationale': rationale,
                    'model_version': self.model_version,
                    'suggested_date': datetime.now().isoformat()
                })
                
            except (ValueError, TypeError) as e:
                logger.warning(f"Error processing suggestion: {e}")
                continue
        
        # Sort by confidence descending
        processed.sort(key=lambda x: x['confidence'], reverse=True)
        
        return processed
    
    def _process_behavioral_suggestion(self, suggestion: Dict) -> Dict[str, Any]:
        """Process and validate behavioral suggestion"""
        try:
            behavioral_score = int(suggestion.get('behavioral_score', 0))
            confidence = float(suggestion.get('confidence', 0.0))
            rationale = suggestion.get('rationale', '').strip()
            
            # Validate behavioral score
            if not (-3 <= behavioral_score <= 3):
                logger.warning(f"Invalid behavioral score: {behavioral_score}")
                behavioral_score = max(-3, min(3, behavioral_score))
            
            # Validate confidence
            if not (0.0 <= confidence <= 1.0):
                logger.warning(f"Invalid confidence score: {confidence}")
                confidence = max(0.0, min(1.0, confidence))
            
            return {
                'behavioral_score': behavioral_score,
                'confidence': confidence,
                'rationale': rationale,
                'scale_name': BEHAVIORAL_SCALE.get(behavioral_score, {}).get('name', 'Unknown'),
                'model_version': self.model_version,
                'suggested_date': datetime.now().isoformat()
            }
            
        except (ValueError, TypeError) as e:
            logger.error(f"Error processing behavioral suggestion: {e}")
            # Return neutral suggestion as fallback
            return {
                'behavioral_score': 0,
                'confidence': 0.5,
                'rationale': 'Error processing AI suggestion - defaulting to neutral',
                'scale_name': BEHAVIORAL_SCALE[0]['name'],
                'model_version': self.model_version,
                'suggested_date': datetime.now().isoformat()
            }
    
    def clear_cache(self):
        """Clear AI response cache"""
        self._values_cache.clear()
        self._behavioral_cache.clear()
        logger.info("AI cache cleared")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        return {
            'values_cache_size': len(self._values_cache),
            'behavioral_cache_size': len(self._behavioral_cache)
        }


class ValuesBehavioralAIWorker:
    """Async worker for AI operations to avoid blocking the UI"""
    
    def __init__(self, ai_service: ValuesBehavioralAI):
        self.ai_service = ai_service
        self.worker_thread = None
    
    def get_values_suggestions_async(self, text_section: str, callback=None, use_cache: bool = True):
        """Get values suggestions asynchronously"""
        def worker():
            try:
                result = self.ai_service.get_values_suggestions_sync(text_section, use_cache)
                if callback:
                    callback(result, None)
            except Exception as e:
                if callback:
                    callback(None, str(e))
        
        from threading import Thread
        self.worker_thread = Thread(target=worker)
        self.worker_thread.daemon = True
        self.worker_thread.start()
    
    def get_behavioral_suggestion_async(self, sentence_text: str, selected_value: str, 
                                      callback=None, use_cache: bool = True):
        """Get behavioral suggestion asynchronously"""
        def worker():
            try:
                result = self.ai_service.get_behavioral_suggestion_sync(
                    sentence_text, selected_value, use_cache
                )
                if callback:
                    callback(result, None)
            except Exception as e:
                if callback:
                    callback(None, str(e))
        
        from threading import Thread
        self.worker_thread = Thread(target=worker)
        self.worker_thread.daemon = True
        self.worker_thread.start()


def create_values_behavioral_ai(app) -> ValuesBehavioralAI:
    """Factory function to create ValuesBehavioralAI instance"""
    return ValuesBehavioralAI(app)