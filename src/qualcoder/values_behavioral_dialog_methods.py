"""
Values and Behavioral Enactment Coder - Dialog Methods (Part 2)
Additional methods for the main values behavioral dialog
This file extends values_behavioral_dialog.py
"""

import logging
from typing import List, Optional
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QTreeWidgetItem, QMessageBox
from PyQt6.QtGui import QFont, QColor

from .values_behavioral_constants import (
    CodingStage, ConfidenceLevel, BEHAVIORAL_SCALE, CONFIDENCE_THRESHOLDS
)
from .values_behavioral_models import (
    ClaudeValuesSuggestion, ClaudeBehavioralSuggestion
)
from .helpers import Message

logger = logging.getLogger(__name__)


# These methods should be added to the ValuesBehavioralDialog class
class ValuesBehavioralDialogMethods:
    """Mixin class containing additional methods for ValuesBehavioralDialog"""
    
    def previous_section(self):
        """Navigate to the previous section"""
        if self.current_section_index > 0:
            self.current_section_index -= 1
            self.display_current_section()
            self.clear_coding_ui()
            
            # Auto-get suggestions if enabled
            if self.auto_suggest_enabled and self.current_stage == CodingStage.VALUES:
                self.get_values_suggestions()
    
    def next_section(self):
        """Navigate to the next section"""
        if self.current_section_index < len(self.document_sections) - 1:
            self.current_section_index += 1
            self.display_current_section()
            self.clear_coding_ui()
            
            # Auto-get suggestions if enabled
            if self.auto_suggest_enabled and self.current_stage == CodingStage.VALUES:
                self.get_values_suggestions()
    
    def previous_sentence(self):
        """Navigate to the previous sentence (Stage 2 only)"""
        if self.current_stage != CodingStage.BEHAVIORAL:
            return
            
        if self.current_sentence_index > 0:
            self.current_sentence_index -= 1
            self.display_current_sentence()
            self.clear_behavioral_ui()
    
    def next_sentence(self):
        """Navigate to the next sentence (Stage 2 only)"""
        if self.current_stage != CodingStage.BEHAVIORAL:
            return
            
        current_sentences = self.get_current_section_sentences()
        if self.current_sentence_index < len(current_sentences) - 1:
            self.current_sentence_index += 1
            self.display_current_sentence()
            self.clear_behavioral_ui()
    
    def get_values_suggestions(self):
        """Get AI suggestions for values coding"""
        if not self.document_sections or self.current_section_index >= len(self.document_sections):
            return
        
        try:
            section = self.document_sections[self.current_section_index]
            
            # Check cache first
            if section.section_id in self.suggestions_cache:
                suggestions = self.suggestions_cache[section.section_id]
            else:
                # Get new suggestions
                self.get_suggestions_btn.setEnabled(False)
                self.get_suggestions_btn.setText("Getting suggestions...")
                
                suggestions = self.service.get_claude_values_suggestions(
                    section.section_id, 
                    section.section_text,
                    use_cache=True
                )
                
                self.suggestions_cache[section.section_id] = suggestions
                
                self.get_suggestions_btn.setEnabled(True)
                self.get_suggestions_btn.setText("Get Suggestions")
            
            # Update suggestions tree
            self.update_suggestions_tree(suggestions)
            
        except Exception as e:
            logger.error(f"Error getting values suggestions: {e}")
            Message(self.app, "Error", f"Failed to get suggestions: {e}", "warning").exec()
            
            self.get_suggestions_btn.setEnabled(True)
            self.get_suggestions_btn.setText("Get Suggestions")
    
    def update_suggestions_tree(self, suggestions: List[ClaudeValuesSuggestion]):
        """Update the suggestions tree widget"""
        self.suggestions_tree.clear()
        
        for suggestion in suggestions:
            # Only show suggestions above threshold
            if suggestion.confidence_score < (self.confidence_threshold):
                continue
            
            item = QTreeWidgetItem(self.suggestions_tree)
            item.setText(0, suggestion.value_name)
            item.setText(1, f"{suggestion.confidence_score:.2%}")
            item.setText(2, self.get_value_category(suggestion.value_id))
            item.setText(3, suggestion.rationale)
            
            # Color code by confidence
            if suggestion.confidence_score >= 0.9:
                item.setBackground(0, QColor("#E8F5E8"))  # Light green
            elif suggestion.confidence_score >= 0.7:
                item.setBackground(0, QColor("#FFF8DC"))  # Light yellow
            else:
                item.setBackground(0, QColor("#FFE4E1"))  # Light pink
            
            # Store suggestion data
            item.setData(0, Qt.ItemDataRole.UserRole, suggestion)
        
        # Expand tree
        self.suggestions_tree.expandAll()
        
        # Auto-select highest confidence suggestion if above threshold
        if suggestions and suggestions[0].confidence_score >= CONFIDENCE_THRESHOLDS['values_auto_accept']:
            self.suggestions_tree.setCurrentItem(self.suggestions_tree.topLevelItem(0))
    
    def suggestion_clicked(self, item: QTreeWidgetItem, column: int):
        """Handle suggestion click"""
        suggestion = item.data(0, Qt.ItemDataRole.UserRole)
        if suggestion:
            # Auto-fill values coding form
            self.select_value_by_id(suggestion.value_id)
            
            # Set confidence based on suggestion confidence
            if suggestion.confidence_score >= 0.9:
                self.confidence_combo.setCurrentText("High")
            elif suggestion.confidence_score >= 0.7:
                self.confidence_combo.setCurrentText("Medium")
            else:
                self.confidence_combo.setCurrentText("Low")
            
            # Add rationale to notes
            current_notes = self.values_notes_text.toPlainText()
            if current_notes:
                current_notes += "\n\n"
            current_notes += f"AI Suggestion ({suggestion.confidence_score:.1%}): {suggestion.rationale}"
            self.values_notes_text.setPlainText(current_notes)
            
            # Enable save button
            self.save_values_btn.setEnabled(True)
    
    def value_selected(self, item: QTreeWidgetItem, column: int):
        """Handle manual value selection"""
        value_id = item.data(0, Qt.ItemDataRole.UserRole)
        if value_id:
            self.select_value_by_id(value_id)
    
    def select_value_by_id(self, value_id: int):
        """Select a value by its ID"""
        # Find and select the value in the tree
        iterator = QtCore.QTreeWidgetItemIterator(self.values_tree)
        while iterator.value():
            item = iterator.value()
            if item.data(0, Qt.ItemDataRole.UserRole) == value_id:
                self.values_tree.setCurrentItem(item)
                break
            iterator += 1
        
        # Clear custom value input
        self.custom_value_input.clear()
        
        # Enable save button
        self.save_values_btn.setEnabled(True)
    
    def get_value_category(self, value_id: int) -> str:
        """Get the category for a value ID"""
        try:
            core_values = self.service.get_core_values()
            for value in core_values:
                if value.value_id == value_id:
                    return value.value_category
            return "Unknown"
        except:
            return "Unknown"
    
    def save_values_coding(self):
        """Save the current values coding"""
        if not self.document_sections or self.current_section_index >= len(self.document_sections):
            return
        
        try:
            section = self.document_sections[self.current_section_index]
            
            # Get selected value or custom value
            value_id = None
            custom_value_name = ""
            
            current_item = self.values_tree.currentItem()
            if current_item and current_item.data(0, Qt.ItemDataRole.UserRole):
                value_id = current_item.data(0, Qt.ItemDataRole.UserRole)
            elif self.custom_value_input.text().strip():
                custom_value_name = self.custom_value_input.text().strip()
            else:
                Message(self.app, "Error", "Please select a value or enter a custom value", "warning").exec()
                return
            
            # Get other form values
            confidence_text = self.confidence_combo.currentText().lower()
            confidence_level = ConfidenceLevel(confidence_text)
            coder_notes = self.values_notes_text.toPlainText()
            coder_name = self.app.settings.get('codername', 'default')
            
            # Check if this was selected from suggestions
            selected_from_suggestion = any(
                item.data(0, Qt.ItemDataRole.UserRole) and 
                (item.data(0, Qt.ItemDataRole.UserRole).value_id == value_id)
                for item in [self.suggestions_tree.itemAt(i, 0) for i in range(self.suggestions_tree.topLevelItemCount())]
                if item and item.data(0, Qt.ItemDataRole.UserRole)
            )
            
            # Save to database
            coding = self.service.save_values_coding(
                section_id=section.section_id,
                value_id=value_id,
                custom_value_name=custom_value_name,
                coder_name=coder_name,
                confidence_level=confidence_level,
                coder_notes=coder_notes,
                selected_from_suggestion=selected_from_suggestion
            )
            
            # Update UI
            self.save_values_btn.setEnabled(False)
            self.lock_values_btn.setEnabled(True)
            
            # Update progress
            self.update_session_ui()
            
            self.status_bar.showMessage("Values coding saved")
            
        except Exception as e:
            logger.error(f"Error saving values coding: {e}")
            Message(self.app, "Error", f"Failed to save values coding: {e}", "critical").exec()
    
    def lock_values_coding(self):
        """Lock the current values coding and move to next"""
        # Implementation for locking values coding
        # This would call the service method and then navigate to next section
        pass
    
    def complete_current_stage(self):
        """Complete the current coding stage"""
        if self.current_stage == CodingStage.VALUES:
            self.complete_values_stage()
        else:
            self.complete_behavioral_stage()
    
    def complete_values_stage(self):
        """Complete Stage 1 and move to Stage 2"""
        try:
            # Verify all sections have values coding
            # Enable Stage 2 tab
            self.coding_tabs.setTabEnabled(1, True)
            self.current_stage = CodingStage.BEHAVIORAL
            
            # Switch to behavioral tab
            self.coding_tabs.setCurrentIndex(1)
            
            # Parse sentences for current section
            self.parse_current_section_sentences()
            
            # Update UI
            self.stage_label.setText("Stage 2: Behavioral Enactment")
            self.sentence_nav_frame.setVisible(True)
            
            # Emit signal
            self.stage_completed.emit(1)
            
            self.status_bar.showMessage("Stage 1 complete - Now coding behavioral enactment")
            
        except Exception as e:
            logger.error(f"Error completing values stage: {e}")
            Message(self.app, "Error", f"Failed to complete values stage: {e}", "critical").exec()
    
    def parse_current_section_sentences(self):
        """Parse the current section into sentences"""
        if not self.document_sections or self.current_section_index >= len(self.document_sections):
            return
        
        try:
            section = self.document_sections[self.current_section_index]
            
            # Check if sentences already parsed
            if section.section_id not in self.document_sentences:
                sentences = self.service.parse_section_sentences(
                    section.section_id, 
                    section.section_text,
                    use_ai=True
                )
                self.document_sentences[section.section_id] = sentences
            
            # Reset sentence index and display first sentence
            self.current_sentence_index = 0
            self.display_current_sentence()
            
        except Exception as e:
            logger.error(f"Error parsing sentences: {e}")
            Message(self.app, "Error", f"Failed to parse sentences: {e}", "warning").exec()
    
    def get_current_section_sentences(self):
        """Get sentences for the current section"""
        if not self.document_sections or self.current_section_index >= len(self.document_sections):
            return []
        
        section = self.document_sections[self.current_section_index]
        return self.document_sentences.get(section.section_id, [])
    
    def display_current_sentence(self):
        """Display the current sentence for behavioral coding"""
        sentences = self.get_current_section_sentences()
        if not sentences or self.current_sentence_index >= len(sentences):
            return
        
        sentence = sentences[self.current_sentence_index]
        
        # Update sentence label
        self.sentence_label.setText(f"Sentence: {self.current_sentence_index + 1} / {len(sentences)}")
        
        # Highlight current sentence in document text
        self.highlight_current_sentence(sentence)
        
        # Update navigation buttons
        self.prev_sentence_btn.setEnabled(self.current_sentence_index > 0)
        self.next_sentence_btn.setEnabled(self.current_sentence_index < len(sentences) - 1)
        
        # Load selected value info
        self.load_current_value_info()
    
    def highlight_current_sentence(self, sentence):
        """Highlight the current sentence in the document view"""
        # Get current section
        section = self.document_sections[self.current_section_index]
        
        # Create highlighted HTML with current sentence emphasized
        sentences = self.get_current_section_sentences()
        html_parts = []
        
        for i, sent in enumerate(sentences):
            if i == self.current_sentence_index:
                html_parts.append(f'<span style="background-color: #FFD700; font-weight: bold; padding: 2px;">{sent.sentence_text}</span>')
            else:
                html_parts.append(sent.sentence_text)
        
        html_text = f"""
        <div style="font-family: 'Noto Sans'; font-size: 12pt; line-height: 1.8;">
            <p style="background-color: #f0f8ff; padding: 15px; border-left: 4px solid #4169e1;">
                {' '.join(html_parts)}
            </p>
        </div>
        """
        
        self.document_text.setHtml(html_text)
    
    def load_current_value_info(self):
        """Load information about the selected value for the current section"""
        # Get the values coding for current section
        # Display value name and definition
        # This would query the database for the locked values coding
        pass
    
    def behavioral_score_changed(self, value: int):
        """Handle behavioral score slider change"""
        self.update_behavioral_scale_description(value)
        self.save_behavioral_btn.setEnabled(True)
    
    def update_behavioral_scale_description(self, score: int):
        """Update the behavioral scale description"""
        scale_info = BEHAVIORAL_SCALE.get(score, {})
        
        # Update current score label
        self.current_score_label.setText(f"Current Score: {score} ({scale_info.get('name', 'Unknown')})")
        
        # Update description text
        description_html = f"""
        <div style="font-family: Arial; font-size: 11pt;">
            <h4 style="color: {'green' if score > 0 else 'red' if score < 0 else 'gray'};">
                {scale_info.get('name', 'Unknown')} ({score})
            </h4>
            <p><strong>Description:</strong> {scale_info.get('full_description', 'No description available')}</p>
            <p><strong>Examples:</strong> {scale_info.get('examples', 'No examples available')}</p>
        </div>
        """
        
        self.scale_description_text.setHtml(description_html)
    
    def get_behavioral_suggestions(self):
        """Get AI suggestions for behavioral coding"""
        # Implementation for getting behavioral suggestions from Claude
        pass
    
    def save_behavioral_coding(self):
        """Save the current behavioral coding"""
        # Implementation for saving behavioral coding
        pass
    
    def lock_behavioral_coding(self):
        """Lock the current behavioral coding"""
        # Implementation for locking behavioral coding
        pass
    
    def complete_behavioral_stage(self):
        """Complete Stage 2 and finish session"""
        # Implementation for completing the behavioral stage
        pass
    
    def tab_changed(self, index: int):
        """Handle tab change between values and behavioral coding"""
        if index == 0:  # Values tab
            self.current_stage = CodingStage.VALUES
            self.sentence_nav_frame.setVisible(False)
        elif index == 1:  # Behavioral tab
            self.current_stage = CodingStage.BEHAVIORAL
            self.sentence_nav_frame.setVisible(True)
            self.parse_current_section_sentences()
    
    def auto_suggest_toggled(self, state: int):
        """Handle auto-suggest checkbox toggle"""
        self.auto_suggest_enabled = state == Qt.CheckState.Checked.value
    
    def confidence_threshold_changed(self, value: int):
        """Handle confidence threshold change"""
        self.confidence_threshold = value / 100.0
    
    def pause_session(self):
        """Pause the current coding session"""
        # Implementation for pausing session
        pass
    
    def clear_coding_ui(self):
        """Clear the coding UI when navigating"""
        # Clear suggestions
        self.suggestions_tree.clear()
        
        # Clear form fields
        self.values_tree.setCurrentItem(None)
        self.custom_value_input.clear()
        self.values_notes_text.clear()
        self.confidence_combo.setCurrentIndex(0)
        
        # Disable buttons
        self.save_values_btn.setEnabled(False)
        self.lock_values_btn.setEnabled(False)
    
    def clear_behavioral_ui(self):
        """Clear the behavioral coding UI when navigating"""
        # Reset slider
        self.behavioral_slider.setValue(0)
        
        # Clear text areas
        self.behavioral_rationale_text.clear()
        self.behavioral_suggestions_text.clear()
        
        # Disable buttons
        self.save_behavioral_btn.setEnabled(False)
        self.lock_behavioral_btn.setEnabled(False)