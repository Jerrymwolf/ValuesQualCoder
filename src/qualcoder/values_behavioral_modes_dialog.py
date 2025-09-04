"""
Values and Behavioral Enactment Coder - Multi-Mode Dialog
Enhanced dialog supporting Open Coding, Taxonomy Coding, and Hybrid modes
"""

import logging
import asyncio
from datetime import datetime
from typing import List, Optional, Dict, Any

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QFont, QColor, QPalette
from PyQt6.QtWidgets import (QVBoxLayout, QHBoxLayout, QSplitter, QTabWidget,
                           QTextEdit, QTreeWidget, QTreeWidgetItem, QPushButton,
                           QLabel, QComboBox, QSpinBox, QSlider, QProgressBar,
                           QGroupBox, QRadioButton, QCheckBox, QTextBrowser,
                           QFrame, QMessageBox, QDialog, QDialogButtonBox,
                           QButtonGroup, QTableWidget, QTableWidgetItem,
                           QHeaderView, QSplitterHandle)

from .values_behavioral_dialog import ValuesBehavioralDialog
from .values_behavioral_open_service import OpenCodingService
from .values_behavioral_constants import CodingMode, BEHAVIORAL_SCALE, VALUES_CATEGORY_COLORS
from .values_behavioral_open_coding import (
    OpenCodedValue, TaxonomyMapping, HybridCodingResult, MappingType
)
from .helpers import Message

logger = logging.getLogger(__name__)


class CodingModeSelectionDialog(QDialog):
    """Dialog for selecting coding mode at session start"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_mode = CodingMode.TAXONOMY
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the mode selection UI"""
        self.setWindowTitle("Select Coding Mode")
        self.setMinimumSize(500, 400)
        
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Choose Your Values Coding Approach")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Description
        desc = QLabel("Select the coding approach that best fits your research methodology:")
        desc.setFont(QFont("Arial", 10))
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc.setWordWrap(True)
        layout.addWidget(desc)
        
        layout.addSpacing(20)
        
        # Mode selection
        self.mode_group = QButtonGroup()
        
        # Open Coding Mode
        open_radio = QRadioButton("Open Coding")
        open_radio.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.mode_group.addButton(open_radio, CodingMode.OPEN.value.__hash__())
        layout.addWidget(open_radio)
        
        open_desc = QLabel("""
• Identify values without constraints using Claude's general knowledge
• Suitable for exploratory research and taxonomy development
• AI suggests any values it identifies in the text
• Best for: Initial exploration, theory building, taxonomy validation
        """.strip())
        open_desc.setFont(QFont("Arial", 9))
        open_desc.setIndent(20)
        open_desc.setWordWrap(True)
        layout.addWidget(open_desc)
        
        layout.addSpacing(10)
        
        # Taxonomy Coding Mode
        taxonomy_radio = QRadioButton("Taxonomy-Based Coding")
        taxonomy_radio.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        taxonomy_radio.setChecked(True)  # Default
        self.mode_group.addButton(taxonomy_radio, CodingMode.TAXONOMY.value.__hash__())
        layout.addWidget(taxonomy_radio)
        
        taxonomy_desc = QLabel("""
• Use predefined 32-value taxonomy from Phase 0 research
• AI suggests values only from the established taxonomy
• Suitable for structured analysis with existing framework
• Best for: Hypothesis testing, comparative studies, standardized analysis
        """.strip())
        taxonomy_desc.setFont(QFont("Arial", 9))
        taxonomy_desc.setIndent(20)
        taxonomy_desc.setWordWrap(True)
        layout.addWidget(taxonomy_desc)
        
        layout.addSpacing(10)
        
        # Hybrid Coding Mode
        hybrid_radio = QRadioButton("Hybrid Coding")
        hybrid_radio.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.mode_group.addButton(hybrid_radio, CodingMode.HYBRID.value.__hash__())
        layout.addWidget(hybrid_radio)
        
        hybrid_desc = QLabel("""
• Combine both approaches: identify values openly AND map to taxonomy
• Shows gaps between open coding and predefined taxonomy
• Suitable for taxonomy validation and refinement research
• Best for: Taxonomy development, validation studies, comprehensive analysis
        """.strip())
        hybrid_desc.setFont(QFont("Arial", 9))
        hybrid_desc.setIndent(20)
        hybrid_desc.setWordWrap(True)
        layout.addWidget(hybrid_desc)
        
        layout.addSpacing(10)
        
        # Validation Mode
        validation_radio = QRadioButton("Taxonomy Validation")
        validation_radio.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.mode_group.addButton(validation_radio, CodingMode.VALIDATION.value.__hash__())
        layout.addWidget(validation_radio)
        
        validation_desc = QLabel("""
• Start with open coding, then systematically compare to taxonomy
• Designed specifically for taxonomy validation research
• Provides detailed mapping analysis and recommendations
• Best for: Your current research goal of validating the 32-value taxonomy
        """.strip())
        validation_desc.setFont(QFont("Arial", 9))
        validation_desc.setIndent(20)
        validation_desc.setWordWrap(True)
        validation_desc.setStyleSheet("QLabel { color: #2E8B57; font-weight: 500; }")
        layout.addWidget(validation_desc)
        
        layout.addStretch()
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        # Connect radio button changes
        self.mode_group.buttonToggled.connect(self.mode_changed)
    
    def mode_changed(self, button, checked):
        """Handle mode selection change"""
        if checked:
            button_id = self.mode_group.id(button)
            if button_id == CodingMode.OPEN.value.__hash__():
                self.selected_mode = CodingMode.OPEN
            elif button_id == CodingMode.TAXONOMY.value.__hash__():
                self.selected_mode = CodingMode.TAXONOMY
            elif button_id == CodingMode.HYBRID.value.__hash__():
                self.selected_mode = CodingMode.HYBRID
            elif button_id == CodingMode.VALIDATION.value.__hash__():
                self.selected_mode = CodingMode.VALIDATION
    
    def get_selected_mode(self) -> CodingMode:
        """Get the selected coding mode"""
        return self.selected_mode


class OpenCodingTab(QtWidgets.QWidget):
    """Tab for open coding interface"""
    
    def __init__(self, parent_dialog):
        super().__init__()
        self.parent_dialog = parent_dialog
        self.open_service = OpenCodingService(parent_dialog.app)
        self.current_open_values: List[OpenCodedValue] = []
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the open coding UI"""
        layout = QVBoxLayout(self)
        
        # Instructions
        instructions = QLabel("Open Coding: Identify any values present in the text without constraints")
        instructions.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        instructions.setStyleSheet("QLabel { color: #2E8B57; padding: 5px; }")
        layout.addWidget(instructions)
        
        # AI suggestions section
        suggestions_group = QGroupBox("AI Open Coding Suggestions")
        suggestions_layout = QVBoxLayout(suggestions_group)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        self.get_open_suggestions_btn = QPushButton("Get Open Coding Suggestions")
        self.get_open_suggestions_btn.clicked.connect(self.get_open_suggestions)
        controls_layout.addWidget(self.get_open_suggestions_btn)
        
        controls_layout.addStretch()
        
        self.confidence_threshold_spin = QSpinBox()
        self.confidence_threshold_spin.setRange(50, 95)
        self.confidence_threshold_spin.setValue(60)
        self.confidence_threshold_spin.setSuffix("%")
        controls_layout.addWidget(QLabel("Min confidence:"))
        controls_layout.addWidget(self.confidence_threshold_spin)
        
        suggestions_layout.addLayout(controls_layout)
        
        # Open coding suggestions table
        self.open_suggestions_table = QTableWidget()
        self.open_suggestions_table.setColumnCount(4)
        self.open_suggestions_table.setHorizontalHeaderLabels(["Value", "Category", "Confidence", "Rationale"])
        
        # Set column widths
        header = self.open_suggestions_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        
        self.open_suggestions_table.setMaximumHeight(200)
        self.open_suggestions_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.open_suggestions_table.cellClicked.connect(self.suggestion_selected)
        suggestions_layout.addWidget(self.open_suggestions_table)
        
        layout.addWidget(suggestions_group)
        
        # Manual value entry
        manual_group = QGroupBox("Manual Value Entry")
        manual_layout = QVBoxLayout(manual_group)
        
        manual_entry_layout = QHBoxLayout()
        manual_entry_layout.addWidget(QLabel("Value:"))
        
        self.manual_value_input = QtWidgets.QLineEdit()
        self.manual_value_input.setPlaceholderText("Enter any value you identify...")
        manual_entry_layout.addWidget(self.manual_value_input)
        
        manual_entry_layout.addWidget(QLabel("Category:"))
        self.manual_category_input = QtWidgets.QLineEdit()
        self.manual_category_input.setPlaceholderText("Suggested category...")
        manual_entry_layout.addWidget(self.manual_category_input)
        
        manual_layout.addLayout(manual_entry_layout)
        
        # Rationale
        manual_layout.addWidget(QLabel("Rationale:"))
        self.manual_rationale_text = QTextEdit()
        self.manual_rationale_text.setMaximumHeight(80)
        self.manual_rationale_text.setPlaceholderText("Explain why you identify this value in the text...")
        manual_layout.addWidget(self.manual_rationale_text)
        
        layout.addWidget(manual_group)
        
        # Selected values display
        selected_group = QGroupBox("Selected Open Values")
        selected_layout = QVBoxLayout(selected_group)
        
        self.selected_values_table = QTableWidget()
        self.selected_values_table.setColumnCount(4)
        self.selected_values_table.setHorizontalHeaderLabels(["Value", "Category", "Confidence", "Rationale"])
        
        header = self.selected_values_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        
        selected_layout.addWidget(self.selected_values_table)
        
        layout.addWidget(selected_group)
        
        # Action buttons
        action_layout = QHBoxLayout()
        
        self.add_value_btn = QPushButton("Add Selected Value")
        self.add_value_btn.clicked.connect(self.add_selected_value)
        self.add_value_btn.setEnabled(False)
        action_layout.addWidget(self.add_value_btn)
        
        self.add_manual_btn = QPushButton("Add Manual Value")
        self.add_manual_btn.clicked.connect(self.add_manual_value)
        action_layout.addWidget(self.add_manual_btn)
        
        self.remove_value_btn = QPushButton("Remove Selected")
        self.remove_value_btn.clicked.connect(self.remove_selected_value)
        action_layout.addWidget(self.remove_value_btn)
        
        action_layout.addStretch()
        
        self.save_open_values_btn = QPushButton("Save Open Coding")
        self.save_open_values_btn.clicked.connect(self.save_open_values)
        self.save_open_values_btn.setStyleSheet("QPushButton { background-color: #2E8B57; color: white; font-weight: bold; }")
        action_layout.addWidget(self.save_open_values_btn)
        
        layout.addLayout(action_layout)
    
    def get_open_suggestions(self):
        """Get AI suggestions for open coding"""
        try:
            if not hasattr(self.parent_dialog, 'current_section') or not self.parent_dialog.current_section:
                Message(self.parent_dialog.app, "Error", "No section selected", "warning").exec()
                return
            
            section = self.parent_dialog.current_section
            
            # Disable button and show loading
            self.get_open_suggestions_btn.setEnabled(False)
            self.get_open_suggestions_btn.setText("Getting suggestions...")
            
            # Create worker thread for async operation
            self.suggestion_worker = OpenCodingSuggestionWorker(
                self.open_service, section.section_id, section.section_text
            )
            self.suggestion_worker.suggestions_ready.connect(self.on_suggestions_ready)
            self.suggestion_worker.error_occurred.connect(self.on_suggestions_error)
            self.suggestion_worker.start()
            
        except Exception as e:
            logger.error(f"Error getting open suggestions: {e}")
            Message(self.parent_dialog.app, "Error", f"Failed to get suggestions: {e}", "critical").exec()
            self.get_open_suggestions_btn.setEnabled(True)
            self.get_open_suggestions_btn.setText("Get Open Coding Suggestions")
    
    def on_suggestions_ready(self, suggestions: List[OpenCodedValue]):
        """Handle received suggestions"""
        self.update_suggestions_table(suggestions)
        self.get_open_suggestions_btn.setEnabled(True)
        self.get_open_suggestions_btn.setText("Get Open Coding Suggestions")
    
    def on_suggestions_error(self, error_msg: str):
        """Handle suggestion error"""
        Message(self.parent_dialog.app, "Error", f"Failed to get suggestions: {error_msg}", "critical").exec()
        self.get_open_suggestions_btn.setEnabled(True)
        self.get_open_suggestions_btn.setText("Get Open Coding Suggestions")
    
    def update_suggestions_table(self, suggestions: List[OpenCodedValue]):
        """Update the suggestions table with new suggestions"""
        threshold = self.confidence_threshold_spin.value() / 100.0
        filtered_suggestions = [s for s in suggestions if s.confidence_score >= threshold]
        
        self.open_suggestions_table.setRowCount(len(filtered_suggestions))
        
        for row, suggestion in enumerate(filtered_suggestions):
            self.open_suggestions_table.setItem(row, 0, QTableWidgetItem(suggestion.value_name))
            self.open_suggestions_table.setItem(row, 1, QTableWidgetItem(suggestion.suggested_category or ""))
            
            # Confidence with color coding
            conf_item = QTableWidgetItem(f"{suggestion.confidence_score:.1%}")
            if suggestion.confidence_score >= 0.8:
                conf_item.setBackground(QColor("#E8F5E8"))
            elif suggestion.confidence_score >= 0.6:
                conf_item.setBackground(QColor("#FFF8DC"))
            else:
                conf_item.setBackground(QColor("#FFE4E1"))
            self.open_suggestions_table.setItem(row, 2, conf_item)
            
            self.open_suggestions_table.setItem(row, 3, QTableWidgetItem(suggestion.rationale))
            
            # Store suggestion object
            self.open_suggestions_table.item(row, 0).setData(Qt.ItemDataRole.UserRole, suggestion)
    
    def suggestion_selected(self, row: int, column: int):
        """Handle suggestion selection"""
        suggestion = self.open_suggestions_table.item(row, 0).data(Qt.ItemDataRole.UserRole)
        if suggestion:
            # Auto-fill manual entry fields
            self.manual_value_input.setText(suggestion.value_name)
            self.manual_category_input.setText(suggestion.suggested_category or "")
            self.manual_rationale_text.setPlainText(suggestion.rationale)
            self.add_value_btn.setEnabled(True)
    
    def add_selected_value(self):
        """Add the currently selected suggestion"""
        current_row = self.open_suggestions_table.currentRow()
        if current_row >= 0:
            suggestion = self.open_suggestions_table.item(current_row, 0).data(Qt.ItemDataRole.UserRole)
            if suggestion:
                self.current_open_values.append(suggestion)
                self.update_selected_values_table()
                self.clear_manual_fields()
    
    def add_manual_value(self):
        """Add a manually entered value"""
        value_name = self.manual_value_input.text().strip()
        if not value_name:
            Message(self.parent_dialog.app, "Error", "Please enter a value name", "warning").exec()
            return
        
        # Create manual open value
        manual_value = OpenCodedValue(
            section_id=self.parent_dialog.current_section.section_id if self.parent_dialog.current_section else 0,
            value_name=value_name,
            suggested_category=self.manual_category_input.text().strip(),
            confidence_score=0.8,  # Default confidence for manual entries
            rationale=self.manual_rationale_text.toPlainText().strip(),
            coded_date=datetime.now(),
            coder_name=self.parent_dialog.app.settings.get('codername', 'Manual'),
            model_version="Manual",
            is_validated=True  # Manual entries are considered validated
        )
        
        self.current_open_values.append(manual_value)
        self.update_selected_values_table()
        self.clear_manual_fields()
    
    def remove_selected_value(self):
        """Remove selected value from the list"""
        current_row = self.selected_values_table.currentRow()
        if current_row >= 0 and current_row < len(self.current_open_values):
            del self.current_open_values[current_row]
            self.update_selected_values_table()
    
    def update_selected_values_table(self):
        """Update the selected values table"""
        self.selected_values_table.setRowCount(len(self.current_open_values))
        
        for row, value in enumerate(self.current_open_values):
            self.selected_values_table.setItem(row, 0, QTableWidgetItem(value.value_name))
            self.selected_values_table.setItem(row, 1, QTableWidgetItem(value.suggested_category or ""))
            self.selected_values_table.setItem(row, 2, QTableWidgetItem(f"{value.confidence_score:.1%}"))
            self.selected_values_table.setItem(row, 3, QTableWidgetItem(value.rationale))
    
    def clear_manual_fields(self):
        """Clear manual entry fields"""
        self.manual_value_input.clear()
        self.manual_category_input.clear()
        self.manual_rationale_text.clear()
        self.add_value_btn.setEnabled(False)
    
    def save_open_values(self):
        """Save the open coded values"""
        if not self.current_open_values:
            Message(self.parent_dialog.app, "Warning", "No values to save", "warning").exec()
            return
        
        try:
            # Save each open value
            for open_value in self.current_open_values:
                self.open_service.save_open_coded_value(open_value)
            
            Message(self.parent_dialog.app, "Success", f"Saved {len(self.current_open_values)} open coded values", "information").exec()
            
            # Clear current values
            self.current_open_values.clear()
            self.update_selected_values_table()
            
        except Exception as e:
            logger.error(f"Error saving open values: {e}")
            Message(self.parent_dialog.app, "Error", f"Failed to save values: {e}", "critical").exec()


class OpenCodingSuggestionWorker(QThread):
    """Worker thread for getting open coding suggestions"""
    
    suggestions_ready = pyqtSignal(list)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, service: OpenCodingService, section_id: int, section_text: str):
        super().__init__()
        self.service = service
        self.section_id = section_id
        self.section_text = section_text
    
    def run(self):
        """Run the worker thread"""
        try:
            # Create event loop for async operations
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                suggestions = loop.run_until_complete(
                    self.service.get_open_coding_suggestions(self.section_id, self.section_text)
                )
                self.suggestions_ready.emit(suggestions)
            finally:
                loop.close()
                
        except Exception as e:
            self.error_occurred.emit(str(e))


class MultiModeValuesBehavioralDialog(ValuesBehavioralDialog):
    """Extended dialog supporting multiple coding modes"""
    
    def __init__(self, app, parent_textedit=None, file_info=None, coding_mode=CodingMode.TAXONOMY):
        super().__init__(app, parent_textedit, file_info)
        self.coding_mode = coding_mode
        self.open_service = OpenCodingService(app)
        self.current_section = None
        self.modify_ui_for_mode()
    
    def modify_ui_for_mode(self):
        """Modify UI based on selected coding mode"""
        if self.coding_mode == CodingMode.OPEN:
            self.add_open_coding_tab()
            self.coding_tabs.setTabEnabled(0, False)  # Disable taxonomy tab
            self.coding_tabs.setCurrentIndex(2)  # Switch to open coding tab
        elif self.coding_mode == CodingMode.HYBRID:
            self.add_open_coding_tab()
            self.add_hybrid_analysis_tab()
        elif self.coding_mode == CodingMode.VALIDATION:
            self.add_open_coding_tab()
            self.add_taxonomy_validation_tab()
            # Start with open coding
            self.coding_tabs.setCurrentIndex(2)
    
    def add_open_coding_tab(self):
        """Add the open coding tab"""
        self.open_coding_tab = OpenCodingTab(self)
        self.coding_tabs.addTab(self.open_coding_tab, "Open Coding")
    
    def add_hybrid_analysis_tab(self):
        """Add hybrid analysis tab"""
        # Placeholder - would implement hybrid analysis interface
        hybrid_tab = QtWidgets.QWidget()
        layout = QVBoxLayout(hybrid_tab)
        layout.addWidget(QLabel("Hybrid Analysis - Coming Soon"))
        self.coding_tabs.addTab(hybrid_tab, "Hybrid Analysis")
    
    def add_taxonomy_validation_tab(self):
        """Add taxonomy validation tab"""
        # Placeholder - would implement validation interface
        validation_tab = QtWidgets.QWidget()
        layout = QVBoxLayout(validation_tab)
        layout.addWidget(QLabel("Taxonomy Validation - Coming Soon"))
        self.coding_tabs.addTab(validation_tab, "Taxonomy Validation")
    
    def display_current_section(self):
        """Override to store current section reference"""
        super().display_current_section()
        
        if self.document_sections and self.current_section_index < len(self.document_sections):
            self.current_section = self.document_sections[self.current_section_index]