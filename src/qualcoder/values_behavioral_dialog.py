"""
Values and Behavioral Enactment Coder - Main Dialog
Main dialog for two-stage values and behavioral coding interface
"""

import logging
import json
from datetime import datetime
from typing import List, Optional, Dict, Any

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QColor, QPalette
from PyQt6.QtWidgets import (QVBoxLayout, QHBoxLayout, QSplitter, QTabWidget,
                           QTextEdit, QTreeWidget, QTreeWidgetItem, QPushButton,
                           QLabel, QComboBox, QSpinBox, QSlider, QProgressBar,
                           QGroupBox, QRadioButton, QCheckBox, QTextBrowser,
                           QFrame, QMessageBox, QDialog, QDialogButtonBox)

from .code_text import DialogCodeText
from .values_behavioral_service import ValuesBehavioralService
from .values_behavioral_models import (
    ValuesCodingSession, DocumentSection, ValuesCoding, DocumentSentence,
    BehavioralCoding, ClaudeValuesSuggestion, ClaudeBehavioralSuggestion
)
from .values_behavioral_constants import (
    CodingStage, SessionStatus, ConfidenceLevel, BEHAVIORAL_SCALE,
    VALUES_CATEGORY_COLORS, CONFIDENCE_THRESHOLDS
)
from .helpers import Message

logger = logging.getLogger(__name__)


class ValuesBehavioralDialog(QtWidgets.QMainWindow):
    """Main dialog for Values and Behavioral Enactment Coding"""
    
    # Signals
    coding_progress_updated = pyqtSignal(dict)
    stage_completed = pyqtSignal(int)
    session_completed = pyqtSignal(int)
    
    def __init__(self, app, parent_textedit=None, file_info=None):
        """Initialize Values Behavioral Dialog
        
        Args:
            app: Main application instance
            parent_textedit: Parent text edit widget (for integration with main interface)
            file_info: Dictionary with file information {'id': int, 'name': str, 'fulltext': str}
        """
        super().__init__()
        self.app = app
        self.parent_textedit = parent_textedit
        self.file_info = file_info or {}
        
        # Initialize services and data
        self.service = ValuesBehavioralService(app)
        self.current_session: Optional[ValuesCodingSession] = None
        self.document_sections: List[DocumentSection] = []
        self.document_sentences: Dict[int, List[DocumentSentence]] = {}
        self.current_section_index = 0
        self.current_sentence_index = 0
        self.current_stage = CodingStage.VALUES
        
        # UI state
        self.suggestions_cache: Dict[int, List] = {}
        self.auto_suggest_enabled = True
        self.confidence_threshold = CONFIDENCE_THRESHOLDS['values_show_suggestion']
        
        self.setup_ui()
        self.setup_connections()
        self.load_initial_data()
    
    def setup_ui(self):
        """Set up the user interface"""
        self.setWindowTitle("Values and Behavioral Enactment Coder")
        self.setMinimumSize(1200, 800)
        
        # Central widget with main layout
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        
        # Header section with session info and controls
        header_frame = self.create_header_section()
        main_layout.addWidget(header_frame)
        
        # Main content splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(main_splitter)
        
        # Left panel - Document text and navigation
        left_panel = self.create_document_panel()
        main_splitter.addWidget(left_panel)
        
        # Right panel - Coding interface
        right_panel = self.create_coding_panel()
        main_splitter.addWidget(right_panel)
        
        # Set splitter proportions
        main_splitter.setSizes([600, 600])
        
        # Status bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready")
        
    def create_header_section(self) -> QFrame:
        """Create the header section with session controls"""
        header_frame = QFrame()
        header_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        header_frame.setMaximumHeight(100)
        
        layout = QHBoxLayout(header_frame)
        
        # Session info group
        session_group = QGroupBox("Session Information")
        session_layout = QHBoxLayout(session_group)
        
        # File info
        self.file_label = QLabel("No file selected")
        self.file_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        session_layout.addWidget(self.file_label)
        
        session_layout.addWidget(QLabel("|"))
        
        # Current stage
        self.stage_label = QLabel("Stage: Values Identification")
        self.stage_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        session_layout.addWidget(self.stage_label)
        
        session_layout.addWidget(QLabel("|"))
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimumWidth(200)
        session_layout.addWidget(self.progress_bar)
        
        session_layout.addStretch()
        layout.addWidget(session_group)
        
        # Control buttons
        button_group = QGroupBox("Controls")
        button_layout = QHBoxLayout(button_group)
        
        self.start_session_btn = QPushButton("Start New Session")
        self.start_session_btn.clicked.connect(self.start_new_session)
        button_layout.addWidget(self.start_session_btn)
        
        self.pause_session_btn = QPushButton("Pause Session")
        self.pause_session_btn.clicked.connect(self.pause_session)
        self.pause_session_btn.setEnabled(False)
        button_layout.addWidget(self.pause_session_btn)
        
        self.complete_stage_btn = QPushButton("Complete Stage")
        self.complete_stage_btn.clicked.connect(self.complete_current_stage)
        self.complete_stage_btn.setEnabled(False)
        button_layout.addWidget(self.complete_stage_btn)
        
        button_layout.addStretch()
        layout.addWidget(button_group)
        
        return header_frame
    
    def create_document_panel(self) -> QWidget:
        """Create the document display and navigation panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Navigation controls
        nav_frame = QFrame()
        nav_layout = QHBoxLayout(nav_frame)
        
        self.section_label = QLabel("Section: 0 / 0")
        nav_layout.addWidget(self.section_label)
        
        nav_layout.addStretch()
        
        self.prev_section_btn = QPushButton("◀ Previous")
        self.prev_section_btn.clicked.connect(self.previous_section)
        nav_layout.addWidget(self.prev_section_btn)
        
        self.next_section_btn = QPushButton("Next ▶")
        self.next_section_btn.clicked.connect(self.next_section)
        nav_layout.addWidget(self.next_section_btn)
        
        layout.addWidget(nav_frame)
        
        # Document text display
        self.document_text = QTextBrowser()
        self.document_text.setMinimumHeight(400)
        self.document_text.setFont(QFont("Noto Sans", 12))
        layout.addWidget(self.document_text)
        
        # Sentence navigation (for Stage 2)
        self.sentence_nav_frame = QFrame()
        sentence_nav_layout = QHBoxLayout(self.sentence_nav_frame)
        
        self.sentence_label = QLabel("Sentence: 0 / 0")
        sentence_nav_layout.addWidget(self.sentence_label)
        
        sentence_nav_layout.addStretch()
        
        self.prev_sentence_btn = QPushButton("◀ Previous")
        self.prev_sentence_btn.clicked.connect(self.previous_sentence)
        sentence_nav_layout.addWidget(self.prev_sentence_btn)
        
        self.next_sentence_btn = QPushButton("Next ▶")
        self.next_sentence_btn.clicked.connect(self.next_sentence)
        sentence_nav_layout.addWidget(self.next_sentence_btn)
        
        layout.addWidget(self.sentence_nav_frame)
        self.sentence_nav_frame.setVisible(False)  # Initially hidden
        
        return panel
    
    def create_coding_panel(self) -> QWidget:
        """Create the coding interface panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Create tab widget for different stages
        self.coding_tabs = QTabWidget()
        
        # Stage 1: Values Coding Tab
        self.values_tab = self.create_values_coding_tab()
        self.coding_tabs.addTab(self.values_tab, "Values Identification")
        
        # Stage 2: Behavioral Coding Tab
        self.behavioral_tab = self.create_behavioral_coding_tab()
        self.coding_tabs.addTab(self.behavioral_tab, "Behavioral Enactment")
        self.coding_tabs.setTabEnabled(1, False)  # Disabled initially
        
        layout.addWidget(self.coding_tabs)
        
        return panel
    
    def create_values_coding_tab(self) -> QWidget:
        """Create the values coding interface"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Claude suggestions section
        suggestions_group = QGroupBox("AI Suggestions")
        suggestions_layout = QVBoxLayout(suggestions_group)
        
        # Auto-suggest controls
        controls_layout = QHBoxLayout()
        
        self.auto_suggest_checkbox = QCheckBox("Auto-suggest values")
        self.auto_suggest_checkbox.setChecked(True)
        controls_layout.addWidget(self.auto_suggest_checkbox)
        
        controls_layout.addWidget(QLabel("Confidence threshold:"))
        self.confidence_spinbox = QSpinBox()
        self.confidence_spinbox.setRange(50, 95)
        self.confidence_spinbox.setValue(int(self.confidence_threshold * 100))
        self.confidence_spinbox.setSuffix("%")
        controls_layout.addWidget(self.confidence_spinbox)
        
        self.get_suggestions_btn = QPushButton("Get Suggestions")
        self.get_suggestions_btn.clicked.connect(self.get_values_suggestions)
        controls_layout.addWidget(self.get_suggestions_btn)
        
        controls_layout.addStretch()
        suggestions_layout.addLayout(controls_layout)
        
        # Suggestions list
        self.suggestions_tree = QTreeWidget()
        self.suggestions_tree.setHeaderLabels(["Value", "Confidence", "Category", "Rationale"])
        self.suggestions_tree.setMaximumHeight(200)
        self.suggestions_tree.itemClicked.connect(self.suggestion_clicked)
        suggestions_layout.addWidget(self.suggestions_tree)
        
        layout.addWidget(suggestions_group)
        
        # Manual selection section
        manual_group = QGroupBox("Manual Value Selection")
        manual_layout = QVBoxLayout(manual_group)
        
        # Value category filter
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Category:"))
        
        self.category_filter = QComboBox()
        self.category_filter.addItem("All Categories", "")
        filter_layout.addWidget(self.category_filter)
        
        filter_layout.addStretch()
        manual_layout.addLayout(filter_layout)
        
        # Values tree
        self.values_tree = QTreeWidget()
        self.values_tree.setHeaderLabels(["Value", "Description"])
        self.values_tree.itemClicked.connect(self.value_selected)
        manual_layout.addWidget(self.values_tree)
        
        # Custom value input
        custom_layout = QHBoxLayout()
        custom_layout.addWidget(QLabel("Custom Value:"))
        
        self.custom_value_input = QtWidgets.QLineEdit()
        self.custom_value_input.setPlaceholderText("Enter custom value name...")
        custom_layout.addWidget(self.custom_value_input)
        
        manual_layout.addLayout(custom_layout)
        
        layout.addWidget(manual_group)
        
        # Coding details section
        details_group = QGroupBox("Coding Details")
        details_layout = QVBoxLayout(details_group)
        
        # Confidence level
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("Confidence:"))
        
        self.confidence_combo = QComboBox()
        self.confidence_combo.addItems(["High", "Medium", "Low"])
        conf_layout.addWidget(self.confidence_combo)
        
        conf_layout.addStretch()
        details_layout.addLayout(conf_layout)
        
        # Notes
        details_layout.addWidget(QLabel("Notes:"))
        self.values_notes_text = QTextEdit()
        self.values_notes_text.setMaximumHeight(80)
        details_layout.addWidget(self.values_notes_text)
        
        layout.addWidget(details_group)
        
        # Action buttons
        action_layout = QHBoxLayout()
        
        self.save_values_btn = QPushButton("Save Values Coding")
        self.save_values_btn.clicked.connect(self.save_values_coding)
        self.save_values_btn.setEnabled(False)
        action_layout.addWidget(self.save_values_btn)
        
        self.lock_values_btn = QPushButton("Lock & Continue")
        self.lock_values_btn.clicked.connect(self.lock_values_coding)
        self.lock_values_btn.setEnabled(False)
        action_layout.addWidget(self.lock_values_btn)
        
        action_layout.addStretch()
        layout.addLayout(action_layout)
        
        return tab
    
    def create_behavioral_coding_tab(self) -> QWidget:
        """Create the behavioral coding interface"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Current value display
        value_group = QGroupBox("Selected Value")
        value_layout = QVBoxLayout(value_group)
        
        self.selected_value_label = QLabel("No value selected")
        self.selected_value_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        value_layout.addWidget(self.selected_value_label)
        
        self.value_definition_text = QTextBrowser()
        self.value_definition_text.setMaximumHeight(80)
        value_layout.addWidget(self.value_definition_text)
        
        layout.addWidget(value_group)
        
        # Behavioral scale section
        scale_group = QGroupBox("Behavioral Scale (-3 to +3)")
        scale_layout = QVBoxLayout(scale_group)
        
        # Scale visualization
        self.behavioral_slider = QSlider(Qt.Orientation.Horizontal)
        self.behavioral_slider.setRange(-3, 3)
        self.behavioral_slider.setValue(0)
        self.behavioral_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.behavioral_slider.setTickInterval(1)
        self.behavioral_slider.valueChanged.connect(self.behavioral_score_changed)
        scale_layout.addWidget(self.behavioral_slider)
        
        # Scale labels
        labels_layout = QHBoxLayout()
        for i in range(-3, 4):
            scale_info = BEHAVIORAL_SCALE.get(i, {})
            label = QLabel(f"{i}\n{scale_info.get('name', '')}")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setFont(QFont("Arial", 8))
            if i == 0:
                label.setStyleSheet("QLabel { color: gray; }")
            elif i > 0:
                label.setStyleSheet("QLabel { color: green; }")
            else:
                label.setStyleSheet("QLabel { color: red; }")
            labels_layout.addWidget(label)
        
        scale_layout.addLayout(labels_layout)
        
        # Current selection display
        self.current_score_label = QLabel("Current Score: 0 (Indifference)")
        self.current_score_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        scale_layout.addWidget(self.current_score_label)
        
        # Scale description
        self.scale_description_text = QTextBrowser()
        self.scale_description_text.setMaximumHeight(100)
        scale_layout.addWidget(self.scale_description_text)
        
        layout.addWidget(scale_group)
        
        # AI suggestions for behavioral coding
        behavioral_suggestions_group = QGroupBox("AI Behavioral Suggestions")
        behavioral_suggestions_layout = QVBoxLayout(behavioral_suggestions_group)
        
        self.get_behavioral_suggestions_btn = QPushButton("Get AI Suggestion")
        self.get_behavioral_suggestions_btn.clicked.connect(self.get_behavioral_suggestions)
        behavioral_suggestions_layout.addWidget(self.get_behavioral_suggestions_btn)
        
        self.behavioral_suggestions_text = QTextBrowser()
        self.behavioral_suggestions_text.setMaximumHeight(80)
        behavioral_suggestions_layout.addWidget(self.behavioral_suggestions_text)
        
        layout.addWidget(behavioral_suggestions_group)
        
        # Coding rationale
        rationale_group = QGroupBox("Coding Rationale")
        rationale_layout = QVBoxLayout(rationale_group)
        
        self.behavioral_rationale_text = QTextEdit()
        self.behavioral_rationale_text.setMaximumHeight(100)
        self.behavioral_rationale_text.setPlaceholderText("Explain your reasoning for this behavioral score...")
        rationale_layout.addWidget(self.behavioral_rationale_text)
        
        layout.addWidget(rationale_group)
        
        # Action buttons
        behavioral_action_layout = QHBoxLayout()
        
        self.save_behavioral_btn = QPushButton("Save Behavioral Coding")
        self.save_behavioral_btn.clicked.connect(self.save_behavioral_coding)
        self.save_behavioral_btn.setEnabled(False)
        behavioral_action_layout.addWidget(self.save_behavioral_btn)
        
        self.lock_behavioral_btn = QPushButton("Lock & Continue")
        self.lock_behavioral_btn.clicked.connect(self.lock_behavioral_coding)
        self.lock_behavioral_btn.setEnabled(False)
        behavioral_action_layout.addWidget(self.lock_behavioral_btn)
        
        behavioral_action_layout.addStretch()
        layout.addLayout(behavioral_action_layout)
        
        return tab
    
    def setup_connections(self):
        """Set up signal connections"""
        # Tab change handling
        self.coding_tabs.currentChanged.connect(self.tab_changed)
        
        # Auto-suggest checkbox
        self.auto_suggest_checkbox.stateChanged.connect(self.auto_suggest_toggled)
        
        # Confidence threshold
        self.confidence_spinbox.valueChanged.connect(self.confidence_threshold_changed)
    
    def load_initial_data(self):
        """Load initial data for the interface"""
        try:
            # Load core values into the values tree
            self.load_values_tree()
            
            # Load category filter
            self.load_category_filter()
            
            # Update file info if provided
            if self.file_info:
                self.file_label.setText(f"File: {self.file_info.get('name', 'Unknown')}")
            
            # Update behavioral scale description
            self.update_behavioral_scale_description(0)
            
        except Exception as e:
            logger.error(f"Error loading initial data: {e}")
            Message(self.app, "Error", f"Failed to load initial data: {e}", "warning").exec()
    
    def load_values_tree(self):
        """Load core values into the tree widget"""
        try:
            core_values = self.service.get_core_values()
            self.values_tree.clear()
            
            # Group by category
            categories = {}
            for value in core_values:
                if value.value_category not in categories:
                    categories[value.value_category] = []
                categories[value.value_category].append(value)
            
            # Add to tree
            for category, values in categories.items():
                category_item = QTreeWidgetItem(self.values_tree, [category, ""])
                category_item.setBackground(0, QColor(VALUES_CATEGORY_COLORS.get(category, "#EEEEEE")))
                category_item.setFont(0, QFont("Arial", 10, QFont.Weight.Bold))
                
                for value in values:
                    value_item = QTreeWidgetItem(category_item, [value.value_name, value.description])
                    value_item.setData(0, Qt.ItemDataRole.UserRole, value.value_id)
            
            self.values_tree.expandAll()
            
        except Exception as e:
            logger.error(f"Error loading values tree: {e}")
    
    def load_category_filter(self):
        """Load category options into the filter combo"""
        try:
            core_values = self.service.get_core_values()
            categories = set(value.value_category for value in core_values)
            
            for category in sorted(categories):
                self.category_filter.addItem(category, category)
                
        except Exception as e:
            logger.error(f"Error loading category filter: {e}")
    
    # Event handlers and main functionality methods will be added in the next part...
    
    def start_new_session(self):
        """Start a new coding session"""
        if not self.file_info:
            Message(self.app, "Error", "No file selected for coding", "warning").exec()
            return
        
        try:
            # Get coder name
            coder_name = self.app.settings.get('codername', 'default')
            
            # Create new session
            self.current_session = self.service.create_coding_session(
                fid=self.file_info['id'],
                coder_name=coder_name,
                stage=CodingStage.VALUES
            )
            
            # Parse document sections
            full_text = self.file_info.get('fulltext', '')
            self.document_sections = self.service.parse_document_sections(
                fid=self.file_info['id'],
                session_id=self.current_session.session_id,
                full_text=full_text,
                use_ai=True
            )
            
            # Update UI
            self.update_session_ui()
            self.load_first_section()
            
            # Enable controls
            self.start_session_btn.setEnabled(False)
            self.pause_session_btn.setEnabled(True)
            self.complete_stage_btn.setEnabled(True)
            
            self.status_bar.showMessage(f"Started coding session {self.current_session.session_id}")
            
        except Exception as e:
            logger.error(f"Error starting session: {e}")
            Message(self.app, "Error", f"Failed to start session: {e}", "critical").exec()
    
    def update_session_ui(self):
        """Update the UI based on current session state"""
        if not self.current_session:
            return
        
        # Update progress bar
        progress = self.service.get_coding_progress(self.current_session.session_id)
        if progress:
            self.progress_bar.setValue(int(progress.overall_progress_percentage))
            self.progress_bar.setFormat(f"{progress.overall_progress_percentage:.1f}%")
        
        # Update stage label
        stage_text = "Stage 1: Values Identification" if self.current_stage == CodingStage.VALUES else "Stage 2: Behavioral Enactment"
        self.stage_label.setText(stage_text)
    
    def load_first_section(self):
        """Load the first section for coding"""
        if not self.document_sections:
            return
        
        self.current_section_index = 0
        self.display_current_section()
        
        # Auto-get suggestions if enabled
        if self.auto_suggest_enabled:
            self.get_values_suggestions()
    
    def display_current_section(self):
        """Display the current section in the text widget"""
        if not self.document_sections or self.current_section_index >= len(self.document_sections):
            return
        
        section = self.document_sections[self.current_section_index]
        
        # Update section label
        self.section_label.setText(f"Section: {self.current_section_index + 1} / {len(self.document_sections)}")
        
        # Display section text with highlighting
        highlighted_text = self.highlight_section_text(section.section_text)
        self.document_text.setHtml(highlighted_text)
        
        # Update navigation buttons
        self.prev_section_btn.setEnabled(self.current_section_index > 0)
        self.next_section_btn.setEnabled(self.current_section_index < len(self.document_sections) - 1)
    
    def highlight_section_text(self, text: str) -> str:
        """Apply highlighting to section text"""
        # Basic HTML formatting for now
        # Could be enhanced with syntax highlighting or value-based highlighting
        html_text = f"""
        <div style="font-family: 'Noto Sans'; font-size: 12pt; line-height: 1.6;">
            <p style="background-color: #f0f8ff; padding: 10px; border-left: 4px solid #4169e1;">
                {text.replace('\n', '</p><p>')}
            </p>
        </div>
        """
        return html_text
    
    # Additional methods for navigation, saving, etc. would be implemented here...
    # This provides the basic structure for the two-stage coding interface