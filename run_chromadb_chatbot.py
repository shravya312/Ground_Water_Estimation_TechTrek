#!/usr/bin/env python3
"""
Launcher script for INGRIS ChromaDB Chatbot
"""

import subprocess
import sys
import os

def main():
    """Launch the Streamlit chatbot"""
    print("🚀 Starting INGRIS Groundwater Chatbot with ChromaDB")
    print("=" * 60)
    
    # Check if required files exist
    required_files = [
        "ingris_chromadb_chatbot.py",
        "ingris_rag_ready_complete.csv"
    ]
    
    for file in required_files:
        if not os.path.exists(file):
            print(f"❌ Required file not found: {file}")
            return
    
    print("✅ All required files found")
    
    # Check if ChromaDB collection exists
    try:
        import chromadb
        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_collection("ingris_groundwater_collection")
        count = collection.count()
        print(f"✅ ChromaDB collection found with {count:,} records")
    except Exception as e:
        print(f"❌ ChromaDB collection error: {e}")
        print("💡 Make sure to run upload_to_chromadb.py first")
        return
    
    # Launch Streamlit
    print("\n🌐 Launching Streamlit app...")
    print("📱 The chatbot will open in your browser")
    print("🔗 URL: http://localhost:8501")
    print("\n" + "=" * 60)
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "ingris_chromadb_chatbot.py",
            "--server.port", "8501",
            "--server.headless", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Chatbot stopped by user")
    except Exception as e:
        print(f"❌ Error launching chatbot: {e}")

if __name__ == "__main__":
    main()
