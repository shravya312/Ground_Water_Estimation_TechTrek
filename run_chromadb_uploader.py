#!/usr/bin/env python3
"""
Launcher script for ChromaDB Smart Upload Tracker
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing requirements...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements_chromadb_uploader.txt"
        ])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def run_streamlit():
    """Run the Streamlit app"""
    print("Starting ChromaDB Smart Upload Tracker...")
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "chromadb_smart_uploader.py",
            "--server.port", "8502",
            "--server.headless", "true"
        ])
    except KeyboardInterrupt:
        print("\nUpload tracker stopped by user")
    except Exception as e:
        print(f"Error running Streamlit: {e}")

if __name__ == "__main__":
    print("ChromaDB Smart Upload Tracker Launcher")
    print("=" * 50)
    
    # Check if requirements are installed
    try:
        import streamlit
        import chromadb
        import sentence_transformers
        print("✅ All required packages are already installed")
    except ImportError:
        print("📦 Installing required packages...")
        if not install_requirements():
            print("❌ Failed to install requirements. Please install manually:")
            print("pip install -r requirements_chromadb_uploader.txt")
            sys.exit(1)
    
    # Run the app
    run_streamlit()
