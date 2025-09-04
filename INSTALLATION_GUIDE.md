# Values and Behavioral Enactment Coder - Installation Guide

## Quick Start

### Prerequisites
- QualCoder 3.8+ installed and working
- Python 3.10+ 
- Claude API access (Anthropic API key)

### Installation Steps

#### 1. Install Dependencies
```bash
# Navigate to your QualCoder directory
cd /Users/jeremiahwolf/ValuesQualCoder/ValuesQualCoder

# Install additional Python packages
pip install anthropic json-repair asyncio
```

#### 2. Database Migration
The new system needs to add tables to your QualCoder database:

```python
# Run this in Python or add to QualCoder startup
from src.qualcoder.values_behavioral_migration import migrate_project_to_values_behavioral

# This will be done automatically when you first use the system
# Or you can run it manually on existing projects
```

#### 3. Configure Claude AI
Set up your Anthropic API key in QualCoder's AI settings:
- Go to **Settings → AI Configuration**  
- Add your Anthropic API key
- Select Claude model (recommended: claude-3-5-sonnet-20241022)

#### 4. Integration with QualCoder

Add this code to QualCoder's main window initialization (in `__main__.py`):

```python
# Add to imports at top of file
from .values_behavioral_integration import setup_values_behavioral_integration

# Add to MainWindow.__init__ method after UI setup
def __init__(self, app):
    # ... existing initialization code ...
    
    # Add Values Behavioral Integration
    self.values_behavioral_integration = setup_values_behavioral_integration(self)
```

## Detailed Installation

### Step 1: Check QualCoder Installation

First, make sure QualCoder is working properly:

```bash
# Navigate to QualCoder directory
cd /Users/jeremiahwolf/ValuesQualCoder/ValuesQualCoder

# Try running QualCoder
python -m src.qualcoder
```

If this doesn't work, you need to install QualCoder first.

### Step 2: Install Additional Dependencies

```bash
# Required packages for the values behavioral coder
pip install anthropic>=0.7.0
pip install json-repair>=0.25.0

# Optional but recommended
pip install pandas>=2.0.0  # For data analysis
pip install openpyxl>=3.1.0  # For Excel export
```

### Step 3: File Integration

Copy all the new files into your QualCoder source directory:

```bash
# Copy Python modules to QualCoder source
cp values_behavioral_*.py src/qualcoder/
cp open_coding_schema_extension.sql ./
```

### Step 4: Modify Main QualCoder Files

You need to make a few small modifications to integrate the system:

#### A. Modify `src/qualcoder/__main__.py`

Add these imports near the top of the file:

```python
from .values_behavioral_integration import setup_values_behavioral_integration
```

Add this code in the `MainWindow.__init__` method:

```python
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, app):
        # ... existing code ...
        
        # Add Values Behavioral Integration (add this near the end of __init__)
        try:
            self.values_behavioral_integration = setup_values_behavioral_integration(self)
            logger.info("Values Behavioral Integration loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Values Behavioral Integration: {e}")
```

#### B. Test Integration

Start QualCoder and look for:
1. **New Menu**: "Values & Behavioral Coding" in the menu bar
2. **Toolbar Button**: Values & Behavioral Coding button in toolbar
3. **No Errors**: Check console for any error messages

### Step 5: First Time Setup

#### Enable AI Features
1. Start QualCoder
2. Go to **File → Settings**
3. Find **AI Configuration** section
4. Check "Enable AI features"
5. Enter your Anthropic API key
6. Select Claude model

#### Create/Migrate Project
1. Open an existing project OR create new project
2. Go to **Values & Behavioral Coding → Enable Values & Behavioral Coding**
3. This will add the necessary database tables
4. Confirm the migration when prompted

## Quick Test

### Test the Installation

1. **Start QualCoder**:
   ```bash
   cd /Users/jeremiahwolf/ValuesQualCoder/ValuesQualCoder
   python -m src.qualcoder
   ```

2. **Check Menu**: Look for "Values & Behavioral Coding" menu

3. **Test AI Connection**:
   - Go to Settings → AI
   - Test your Claude API connection

4. **Create Test Project**:
   - File → New Project
   - Add a sample text file
   - Try Values & Behavioral Coding → Start Values & Behavioral Coding

### Expected Behavior

When working correctly, you should see:
- Mode selection dialog when starting coding
- AI suggestions for values
- Different tabs for different coding approaches
- Progress tracking and session management

## Troubleshooting

### Common Issues

#### "AI service not available"
- Check API key in Settings
- Verify internet connection
- Check Anthropic service status

#### "No values behavioral tables found"  
- Run the migration: Values & Behavioral Coding → Enable Values & Behavioral Coding
- Check database file permissions

#### Import errors
- Make sure all Python files are in `src/qualcoder/` directory
- Check Python path and dependencies

#### Permission errors
- Check file/folder permissions
- Make sure QualCoder can write to project directory

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

This will show detailed information about what's happening.

## Alternative Quick Start Script

If you want to automate the installation, create this script:

```python
#!/usr/bin/env python3
"""
Quick setup script for Values Behavioral Coder
"""
import os
import sys
import subprocess

def main():
    print("Setting up Values and Behavioral Enactment Coder...")
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("Error: Python 3.10+ required")
        return False
    
    # Install dependencies
    print("Installing dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "anthropic", "json-repair"])
    
    # Check if QualCoder is available
    try:
        import src.qualcoder
        print("✓ QualCoder found")
    except ImportError:
        print("✗ QualCoder not found. Please install QualCoder first.")
        return False
    
    # Test imports
    try:
        from src.qualcoder.values_behavioral_constants import CodingMode
        from src.qualcoder.values_behavioral_service import ValuesBehavioralService
        print("✓ Values Behavioral modules loaded")
    except ImportError as e:
        print(f"✗ Error loading modules: {e}")
        return False
    
    print("\n✓ Setup complete!")
    print("\nNext steps:")
    print("1. Set up your Anthropic API key in QualCoder Settings")
    print("2. Create or open a project") 
    print("3. Go to Values & Behavioral Coding menu")
    print("4. Choose 'Enable Values & Behavioral Coding' for existing projects")
    print("5. Start coding!")
    
    return True

if __name__ == "__main__":
    main()
```

Save this as `setup_values_behavioral.py` and run:
```bash
python setup_values_behavioral.py
```

This should get you up and running quickly!