#!/usr/bin/env python3
"""
Launcher for Complete INGRIS Upload Tracker
"""

import subprocess
import sys
import os

def main():
    """Launch the complete upload tracker."""
    print("Starting Smart INGRIS Complete Upload Tracker...")
    print("=" * 50)
    
    # Check if required files exist
    required_files = [
        "ingris_rag_ready_complete.csv",
        "smart_upload_tracker_complete.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nPlease ensure all required files are present.")
        return
    
    print("✅ All required files found")
    print("🚀 Launching Streamlit app...")
    
    try:
        # Launch Streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "smart_upload_tracker_complete.py",
            "--server.port", "8502",
            "--server.headless", "true"
        ])
    except KeyboardInterrupt:
        print("\n👋 Upload tracker stopped by user")
    except Exception as e:
        print(f"❌ Error launching upload tracker: {e}")

if __name__ == "__main__":
    main()
