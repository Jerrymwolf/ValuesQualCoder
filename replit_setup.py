"""
Replit Setup for Values and Behavioral Enactment Coder
Simplified version for cloud deployment
"""

import os
import subprocess
import sys

def install_dependencies():
    """Install required packages"""
    packages = [
        'anthropic',
        'json-repair', 
        'PyQt6',
        'sqlite3'  # Usually included with Python
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ Installed {package}")
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {package}")

def create_sample_data():
    """Create sample data for testing"""
    sample_text = """
    I believe strongly in integrity and doing the right thing, even when no one is watching. 
    In my previous role, I had to make a difficult decision when I discovered some financial 
    irregularities. Rather than ignore it or pass it up the chain, I took personal responsibility 
    to investigate and address the issue directly with my team.
    
    Excellence is another core value that drives everything I do. I'm not satisfied with 
    'good enough' - I always push myself and my team to deliver the highest quality work. 
    This sometimes means working late or going the extra mile, but I believe it's worth it 
    to maintain our standards.
    
    I also deeply value helping others and collaboration. When a colleague was struggling 
    with a complex project, I volunteered my own time to mentor them and share my expertise. 
    I find great satisfaction in seeing others succeed and grow in their roles.
    """
    
    return {
        'id': 1,
        'name': 'Sample BEI Interview',
        'fulltext': sample_text.strip()
    }

def main():
    print("=" * 60)
    print("VALUES AND BEHAVIORAL ENACTMENT CODER")
    print("Replit Setup and Demo")
    print("=" * 60)
    
    print("\n1. Installing dependencies...")
    install_dependencies()
    
    print("\n2. Creating sample data...")
    sample_data = create_sample_data()
    print(f"✓ Sample document created: '{sample_data['name']}'")
    
    print("\n3. Testing core functionality...")
    
    # Test constants
    try:
        from values_behavioral_constants import CodingMode, BEHAVIORAL_SCALE
        print("✓ Constants loaded")
        print(f"  - Available modes: {[mode.value for mode in CodingMode]}")
        print(f"  - Behavioral scale range: {min(BEHAVIORAL_SCALE.keys())} to {max(BEHAVIORAL_SCALE.keys())}")
    except ImportError as e:
        print(f"✗ Error loading constants: {e}")
        return False
    
    # Test models
    try:
        from values_behavioral_models import OpenCodedValue, ValuesCoding
        print("✓ Models loaded")
    except ImportError as e:
        print(f"✗ Error loading models: {e}")
        return False
    
    print("\n4. Ready for testing!")
    print("\nNext steps:")
    print("- Set your Anthropic API key as environment variable: ANTHROPIC_API_KEY")
    print("- Run the demo functions to test values coding")
    print("- Modify the sample text with your own interview data")
    
    return True

if __name__ == "__main__":
    main()