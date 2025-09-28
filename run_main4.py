#!/usr/bin/env python3
"""
Launcher for main4.py - ChromaDB Groundwater API
"""

import subprocess
import sys
import os
import time

def check_dependencies():
    """Check if required packages are installed"""
    print("Checking dependencies...")
    
    required_packages = [
        'fastapi',
        'uvicorn',
        'chromadb',
        'sentence-transformers',
        'pandas',
        'requests'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install"
            ] + missing_packages)
            print("✓ All packages installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install packages: {e}")
            return False
    else:
        print("✓ All required packages are available")
        return True

def check_data_files():
    """Check if required data files exist"""
    print("\nChecking data files...")
    
    required_files = [
        "ingris_rag_ready_complete.csv",
        "./chroma_db"
    ]
    
    all_exist = True
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} (missing)")
            all_exist = False
    
    if not all_exist:
        print("\nWarning: Some data files are missing. The API will work with available data.")
    
    return True

def start_api():
    """Start the main4.py API"""
    print("\nStarting Groundwater Estimation API - ChromaDB Version")
    print("=" * 60)
    
    try:
        # Start the API
        subprocess.run([
            sys.executable, "main4.py"
        ])
    except KeyboardInterrupt:
        print("\n[INFO] API stopped by user")
    except Exception as e:
        print(f"[ERROR] Failed to start API: {e}")

def main():
    print("Groundwater Estimation API - ChromaDB Launcher")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        print("Please install missing dependencies and try again.")
        sys.exit(1)
    
    # Check data files
    check_data_files()
    
    # Start API
    start_api()

if __name__ == "__main__":
    main()
