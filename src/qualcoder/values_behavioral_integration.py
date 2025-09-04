"""
Values and Behavioral Enactment Coder - Integration Module
Integrates the two-stage coding system with the existing QualCoder interface
"""

import logging
from typing import Optional, Dict, Any
from PyQt6 import QtWidgets, QtCore
from PyQt6.QtWidgets import QAction, QMenu, QMessageBox

from .values_behavioral_dialog import ValuesBehavioralDialog
from .values_behavioral_migration import migrate_project_to_values_behavioral
from .values_behavioral_service import ValuesBehavioralService
from .helpers import Message

logger = logging.getLogger(__name__)


class ValuesBehavioralIntegration:
    """Handles integration of values behavioral coding with main QualCoder interface"""
    
    def __init__(self, main_window):
        """Initialize integration with main window
        
        Args:
            main_window: QualCoder main window instance
        """
        self.main_window = main_window
        self.app = main_window.app
        self.values_behavioral_dialog: Optional[ValuesBehavioralDialog] = None
        
    def setup_menu_integration(self):
        """Add Values Behavioral Coding options to the main menu"""
        try:
            # Create Values Behavioral menu
            values_menu = QMenu("Values & Behavioral Coding", self.main_window)
            
            # Start Values Coding action
            start_coding_action = QAction("Start Values & Behavioral Coding", self.main_window)
            start_coding_action.setStatusTip("Start two-stage values and behavioral coding")
            start_coding_action.triggered.connect(self.start_values_behavioral_coding)
            values_menu.addAction(start_coding_action)
            
            values_menu.addSeparator()
            
            # Resume Coding Session action
            resume_session_action = QAction("Resume Coding Session", self.main_window)
            resume_session_action.setStatusTip("Resume an existing coding session")
            resume_session_action.triggered.connect(self.resume_coding_session)
            values_menu.addAction(resume_session_action)
            
            # View Progress action
            view_progress_action = QAction("View Coding Progress", self.main_window)
            view_progress_action.setStatusTip("View progress of coding sessions")
            view_progress_action.triggered.connect(self.view_coding_progress)
            values_menu.addAction(view_progress_action)
            
            values_menu.addSeparator()
            
            # Export Results action
            export_results_action = QAction("Export Values & Behavioral Analysis", self.main_window)
            export_results_action.setStatusTip("Export values and behavioral coding results")
            export_results_action.triggered.connect(self.export_results)
            values_menu.addAction(export_results_action)
            
            values_menu.addSeparator()
            
            # Migration action
            migrate_project_action = QAction("Enable Values & Behavioral Coding", self.main_window)
            migrate_project_action.setStatusTip("Enable values and behavioral coding for this project")
            migrate_project_action.triggered.connect(self.migrate_project)
            values_menu.addAction(migrate_project_action)
            
            # Add to menubar (after Codes menu)
            menubar = self.main_window.menuBar()
            actions = menubar.actions()
            
            # Find the position after the "Codes" menu
            insert_position = None
            for i, action in enumerate(actions):
                if action.text() == "Codes" or action.text() == "&Codes":
                    insert_position = i + 1
                    break
            
            if insert_position is not None:
                menubar.insertMenu(actions[insert_position], values_menu)
            else:
                menubar.addMenu(values_menu)
                
            logger.info("Values Behavioral menu integration completed")
            
        except Exception as e:
            logger.error(f"Error setting up menu integration: {e}")
    
    def setup_toolbar_integration(self):
        """Add Values Behavioral Coding buttons to toolbar"""
        try:
            # Find the main toolbar
            toolbar = None
            for child in self.main_window.children():
                if isinstance(child, QtWidgets.QToolBar):
                    toolbar = child
                    break
            
            if not toolbar:
                logger.warning("No toolbar found for integration")
                return
            
            # Add separator
            toolbar.addSeparator()
            
            # Add Values Behavioral Coding action
            values_action = QtWidgets.QAction("Values & Behavioral Coding", self.main_window)
            values_action.setStatusTip("Start Values and Behavioral Coding")
            values_action.triggered.connect(self.start_values_behavioral_coding)
            # You could add an icon here: values_action.setIcon(QIcon("path/to/icon"))
            toolbar.addAction(values_action)
            
            logger.info("Values Behavioral toolbar integration completed")
            
        except Exception as e:
            logger.error(f"Error setting up toolbar integration: {e}")
    
    def setup_context_menu_integration(self):
        """Add Values Behavioral options to context menus"""
        try:
            # This would typically be called when setting up context menus
            # for file lists, text selections, etc.
            pass
        except Exception as e:
            logger.error(f"Error setting up context menu integration: {e}")
    
    def start_values_behavioral_coding(self):
        """Start values and behavioral coding interface"""
        try:
            # Check if project has values behavioral support
            if not self.check_project_compatibility():
                return
            
            # Get current file or let user select
            file_info = self.get_selected_file()
            if not file_info:
                return
            
            # Create and show values behavioral dialog
            self.values_behavioral_dialog = ValuesBehavioralDialog(
                app=self.app,
                parent_textedit=self.main_window.ui.textEdit,
                file_info=file_info
            )
            
            # Connect signals
            self.values_behavioral_dialog.coding_progress_updated.connect(self.on_progress_updated)
            self.values_behavioral_dialog.stage_completed.connect(self.on_stage_completed)
            self.values_behavioral_dialog.session_completed.connect(self.on_session_completed)
            
            # Show dialog
            self.values_behavioral_dialog.show()
            self.values_behavioral_dialog.raise_()
            self.values_behavioral_dialog.activateWindow()
            
            logger.info("Started Values Behavioral Coding interface")
            
        except Exception as e:
            logger.error(f"Error starting values behavioral coding: {e}")
            Message(self.app, "Error", f"Failed to start values behavioral coding: {e}", "critical").exec()
    
    def resume_coding_session(self):
        """Resume an existing coding session"""
        try:
            # Check project compatibility
            if not self.check_project_compatibility():
                return
            
            # Get existing sessions and let user select
            service = ValuesBehavioralService(self.app)
            sessions = service.get_coding_sessions()
            
            if not sessions:
                Message(self.app, "Info", "No existing coding sessions found", "information").exec()
                return
            
            # Show session selection dialog
            session = self.select_coding_session(sessions)
            if not session:
                return
            
            # Get file info for the session
            file_info = self.get_file_info_by_id(session.fid)
            if not file_info:
                Message(self.app, "Error", "File not found for coding session", "warning").exec()
                return
            
            # Create dialog and load session
            self.values_behavioral_dialog = ValuesBehavioralDialog(
                app=self.app,
                parent_textedit=self.main_window.ui.textEdit,
                file_info=file_info
            )
            
            # Load the specific session
            self.values_behavioral_dialog.load_existing_session(session)
            
            # Show dialog
            self.values_behavioral_dialog.show()
            self.values_behavioral_dialog.raise_()
            
            logger.info(f"Resumed coding session {session.session_id}")
            
        except Exception as e:
            logger.error(f"Error resuming coding session: {e}")
            Message(self.app, "Error", f"Failed to resume coding session: {e}", "critical").exec()
    
    def view_coding_progress(self):
        """View coding progress across all sessions"""
        try:
            # Check project compatibility
            if not self.check_project_compatibility():
                return
            
            # Create progress viewer dialog
            from .values_behavioral_progress_dialog import ValuesBehavioralProgressDialog
            progress_dialog = ValuesBehavioralProgressDialog(self.app)
            progress_dialog.exec()
            
        except Exception as e:
            logger.error(f"Error viewing coding progress: {e}")
            Message(self.app, "Error", f"Failed to view coding progress: {e}", "critical").exec()
    
    def export_results(self):
        """Export values and behavioral coding results"""
        try:
            # Check project compatibility
            if not self.check_project_compatibility():
                return
            
            # Create export dialog
            from .values_behavioral_export_dialog import ValuesBehavioralExportDialog
            export_dialog = ValuesBehavioralExportDialog(self.app)
            export_dialog.exec()
            
        except Exception as e:
            logger.error(f"Error exporting results: {e}")
            Message(self.app, "Error", f"Failed to export results: {e}", "critical").exec()
    
    def migrate_project(self):
        """Migrate project to support values behavioral coding"""
        try:
            # Confirm migration
            reply = QMessageBox.question(
                self.main_window,
                "Enable Values & Behavioral Coding",
                "This will add Values and Behavioral Coding support to your project.\n"
                "This operation will modify your project database.\n\n"
                "Do you want to continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply != QMessageBox.StandardButton.Yes:
                return
            
            # Perform migration
            success = migrate_project_to_values_behavioral(self.app)
            
            if success:
                Message(self.app, "Success", 
                       "Values and Behavioral Coding has been enabled for this project!", 
                       "information").exec()
                logger.info("Project migration completed successfully")
            else:
                Message(self.app, "Error", 
                       "Failed to enable Values and Behavioral Coding. Please check the logs.", 
                       "critical").exec()
                
        except Exception as e:
            logger.error(f"Error migrating project: {e}")
            Message(self.app, "Error", f"Failed to migrate project: {e}", "critical").exec()
    
    def check_project_compatibility(self) -> bool:
        """Check if the current project supports values behavioral coding"""
        try:
            cursor = self.app.conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='core_values_taxonomy'")
            result = cursor.fetchone()
            
            if not result:
                reply = QMessageBox.question(
                    self.main_window,
                    "Values & Behavioral Coding Not Enabled",
                    "This project doesn't have Values and Behavioral Coding enabled.\n\n"
                    "Would you like to enable it now?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes
                )
                
                if reply == QMessageBox.StandardButton.Yes:
                    self.migrate_project()
                    # Re-check after migration
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='core_values_taxonomy'")
                    result = cursor.fetchone()
                    return result is not None
                else:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking project compatibility: {e}")
            return False
    
    def get_selected_file(self) -> Optional[Dict[str, Any]]:
        """Get currently selected file or prompt user to select one"""
        try:
            # Try to get current file from main interface
            # This would depend on the main interface structure
            
            # For now, show file selection dialog
            from .select_items import DialogSelectItems
            file_dialog = DialogSelectItems(self.app, "Select file for Values Behavioral Coding", "single")
            
            if file_dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
                selected_files = file_dialog.get_selected()
                if selected_files:
                    file_data = selected_files[0]
                    return {
                        'id': file_data['id'],
                        'name': file_data['name'],
                        'fulltext': file_data.get('fulltext', '')
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting selected file: {e}")
            return None
    
    def get_file_info_by_id(self, file_id: int) -> Optional[Dict[str, Any]]:
        """Get file information by ID"""
        try:
            cursor = self.app.conn.cursor()
            cursor.execute("SELECT id, name, fulltext FROM source WHERE id = ?", (file_id,))
            row = cursor.fetchone()
            
            if row:
                return {
                    'id': row[0],
                    'name': row[1],
                    'fulltext': row[2] or ''
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting file info: {e}")
            return None
    
    def select_coding_session(self, sessions) -> Optional[Any]:
        """Show dialog to select a coding session"""
        try:
            # Create simple selection dialog
            dialog = QtWidgets.QDialog(self.main_window)
            dialog.setWindowTitle("Select Coding Session")
            dialog.setMinimumSize(600, 400)
            
            layout = QtWidgets.QVBoxLayout(dialog)
            
            # Sessions list
            sessions_list = QtWidgets.QListWidget()
            for session in sessions:
                file_name = self.get_file_name_by_id(session.fid)
                item_text = f"Session {session.session_id} - {file_name} ({session.status.value})"
                sessions_list.addItem(item_text)
            
            layout.addWidget(sessions_list)
            
            # Buttons
            buttons = QtWidgets.QDialogButtonBox(
                QtWidgets.QDialogButtonBox.StandardButton.Ok | 
                QtWidgets.QDialogButtonBox.StandardButton.Cancel
            )
            buttons.accepted.connect(dialog.accept)
            buttons.rejected.connect(dialog.reject)
            layout.addWidget(buttons)
            
            if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
                current_row = sessions_list.currentRow()
                if current_row >= 0:
                    return sessions[current_row]
            
            return None
            
        except Exception as e:
            logger.error(f"Error selecting coding session: {e}")
            return None
    
    def get_file_name_by_id(self, file_id: int) -> str:
        """Get file name by ID"""
        try:
            cursor = self.app.conn.cursor()
            cursor.execute("SELECT name FROM source WHERE id = ?", (file_id,))
            row = cursor.fetchone()
            return row[0] if row else f"File {file_id}"
        except Exception as e:
            logger.error(f"Error getting file name: {e}")
            return f"File {file_id}"
    
    # Signal handlers
    def on_progress_updated(self, progress_data: Dict[str, Any]):
        """Handle coding progress updates"""
        # Update main window status or progress indicators
        self.main_window.statusBar().showMessage(
            f"Coding Progress: {progress_data.get('overall_progress_percentage', 0):.1f}%"
        )
    
    def on_stage_completed(self, stage: int):
        """Handle stage completion"""
        stage_name = "Values Identification" if stage == 1 else "Behavioral Enactment"
        self.main_window.ui.textEdit.append(f"Completed Stage {stage}: {stage_name}")
    
    def on_session_completed(self, session_id: int):
        """Handle session completion"""
        self.main_window.ui.textEdit.append(f"Completed coding session {session_id}")
        
        # Refresh any relevant displays in main window
        # This might trigger updates to reports, file lists, etc.
        
    def cleanup(self):
        """Clean up resources when main window closes"""
        if self.values_behavioral_dialog:
            self.values_behavioral_dialog.close()
            self.values_behavioral_dialog = None


def setup_values_behavioral_integration(main_window) -> ValuesBehavioralIntegration:
    """Set up values behavioral coding integration with main QualCoder interface
    
    Args:
        main_window: QualCoder main window instance
        
    Returns:
        ValuesBehavioralIntegration instance
    """
    integration = ValuesBehavioralIntegration(main_window)
    
    try:
        integration.setup_menu_integration()
        integration.setup_toolbar_integration()
        integration.setup_context_menu_integration()
        
        logger.info("Values Behavioral Integration setup completed")
        return integration
        
    except Exception as e:
        logger.error(f"Error setting up Values Behavioral Integration: {e}")
        return integration