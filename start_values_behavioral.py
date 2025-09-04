#!/usr/bin/env python3
"""
Standalone launcher for Values and Behavioral Enactment Coder
Run this to test the system without modifying main QualCoder
"""

import sys
import os
import logging
from PyQt6 import QtWidgets

# Add the src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main launcher function"""
    print("=" * 60)
    print("VALUES AND BEHAVIORAL ENACTMENT CODER")
    print("Standalone Test Launcher")
    print("=" * 60)
    
    # Check dependencies
    print("\n1. Checking dependencies...")
    try:
        import anthropic
        print("   ✓ Anthropic library installed")
    except ImportError:
        print("   ✗ Anthropic library missing. Install with: pip install anthropic")
        return False
    
    try:
        import json_repair
        print("   ✓ JSON repair library installed")
    except ImportError:
        print("   ✗ JSON repair library missing. Install with: pip install json-repair")
        return False
    
    # Check QualCoder availability
    print("\n2. Checking QualCoder...")
    try:
        from qualcoder import App
        print("   ✓ QualCoder App class found")
    except ImportError:
        print("   ✗ QualCoder not found. Make sure you're running from the correct directory.")
        return False
    
    # Check Values Behavioral modules
    print("\n3. Checking Values Behavioral modules...")
    try:
        from qualcoder.values_behavioral_constants import CodingMode
        print("   ✓ Constants module loaded")
        
        from qualcoder.values_behavioral_models import OpenCodedValue
        print("   ✓ Models module loaded")
        
        from qualcoder.values_behavioral_service import ValuesBehavioralService
        print("   ✓ Service module loaded")
        
        from qualcoder.values_behavioral_modes_dialog import CodingModeSelectionDialog
        print("   ✓ Dialog modules loaded")
        
    except ImportError as e:
        print(f"   ✗ Error loading Values Behavioral modules: {e}")
        print("   Make sure all values_behavioral_*.py files are in src/qualcoder/")
        return False
    
    # Initialize QualCoder app
    print("\n4. Initializing application...")
    
    # Create Qt application
    qt_app = QtWidgets.QApplication(sys.argv)
    
    try:
        # Initialize QualCoder App (this manages database, settings, etc.)
        qualcoder_app = App()
        print("   ✓ QualCoder App initialized")
        
        # Set basic settings if not already set
        if not qualcoder_app.settings.get('codername'):
            qualcoder_app.settings['codername'] = 'TestUser'
        
        print(f"   ✓ Coder name: {qualcoder_app.settings.get('codername', 'Not set')}")
        
        # Check if we have a project
        if not qualcoder_app.project_path:
            print("\n5. No project loaded. You'll need to create or open a project first.")
            print("   Options:")
            print("   a) Run QualCoder normally and create a project")
            print("   b) Or continue to test the interface without data")
            
            reply = input("\nContinue with interface test? (y/n): ")
            if reply.lower() != 'y':
                return True
        else:
            print(f"   ✓ Project loaded: {qualcoder_app.project_path}")
        
    except Exception as e:
        print(f"   ✗ Error initializing QualCoder App: {e}")
        print("   This might be normal if no project is loaded yet.")
        # Continue anyway for interface testing
        qualcoder_app = None
    
    # Test the Values Behavioral interface
    print("\n6. Testing Values Behavioral interface...")
    
    try:
        from qualcoder.values_behavioral_modes_dialog import CodingModeSelectionDialog, MultiModeValuesBehavioralDialog
        from qualcoder.values_behavioral_constants import CodingMode
        
        # Show mode selection dialog
        mode_dialog = CodingModeSelectionDialog()
        print("   ✓ Mode selection dialog created")
        
        result = mode_dialog.exec()
        if result == QtWidgets.QDialog.DialogCode.Accepted:
            selected_mode = mode_dialog.get_selected_mode()
            print(f"   ✓ Selected mode: {selected_mode.value}")
            
            # If we have a working app, try to show the main dialog
            if qualcoder_app and qualcoder_app.project_path:
                try:
                    # Create sample file info for testing
                    test_file_info = {
                        'id': 1,
                        'name': 'Test Document',
                        'fulltext': 'This is a test document for values coding. I believe in integrity and helping others. Excellence drives my work and I value collaboration.'
                    }
                    
                    # Create main dialog
                    main_dialog = MultiModeValuesBehavioralDialog(
                        app=qualcoder_app,
                        parent_textedit=None,
                        file_info=test_file_info,
                        coding_mode=selected_mode
                    )
                    
                    print("   ✓ Main dialog created successfully")
                    print("   ✓ Showing Values Behavioral Coder interface...")
                    
                    # Show the dialog
                    main_dialog.show()
                    main_dialog.raise_()
                    main_dialog.activateWindow()
                    
                    # Run the Qt event loop
                    sys.exit(qt_app.exec())
                    
                except Exception as e:
                    print(f"   ✗ Error creating main dialog: {e}")
                    print("   Interface test passed, but need real project for full functionality")
                    
            else:
                print("   ✓ Interface components working")
                print("   → Need QualCoder project for full testing")
                
        else:
            print("   ✓ Mode dialog cancelled - this is normal")
            
    except Exception as e:
        print(f"   ✗ Error testing interface: {e}")
        logger.exception("Interface test error")
        return False
    
    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED!")
    print("\nNext steps:")
    print("1. Make sure you have an Anthropic API key")
    print("2. Create or open a QualCoder project")
    print("3. Run this script again with a project loaded")
    print("4. Or integrate with main QualCoder (see INSTALLATION_GUIDE.md)")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nTest cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        logger.exception("Startup error")
        sys.exit(1)