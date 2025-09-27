#!/usr/bin/env python3
"""
Launch script for the Groundwater Chatbot System
"""

import subprocess
import sys
import time
import os
import webbrowser
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'fastapi', 'uvicorn', 'pandas', 'numpy', 'sentence-transformers',
        'google-generativeai', 'qdrant-client', 'chromadb', 'streamlit'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ All dependencies are installed!")
    return True

def check_environment():
    """Check environment variables"""
    print("\n🔍 Checking environment variables...")
    
    required_vars = ['GEMINI_API_KEY']
    optional_vars = ['QDRANT_URL', 'QDRANT_API_KEY']
    
    missing_required = []
    for var in required_vars:
        if not os.getenv(var):
            missing_required.append(var)
            print(f"❌ {var} (required)")
        else:
            print(f"✅ {var}")
    
    for var in optional_vars:
        if not os.getenv(var):
            print(f"⚠️ {var} (optional - will use fallback)")
        else:
            print(f"✅ {var}")
    
    if missing_required:
        print(f"\n⚠️ Missing required environment variables: {', '.join(missing_required)}")
        print("Create a .env file with these variables")
        return False
    
    return True

def start_backend():
    """Start the backend server"""
    print("\n🚀 Starting backend server...")
    try:
        # Start the FastAPI server
        process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", "main2:app", 
            "--host", "0.0.0.0", "--port", "8000", "--reload"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        print("✅ Backend server starting on http://localhost:8000")
        print("📚 API docs available at http://localhost:8000/docs")
        
        return process
    except Exception as e:
        print(f"❌ Failed to start backend: {e}")
        return None

def test_api():
    """Test if the API is working"""
    print("\n🧪 Testing API...")
    try:
        import requests
        response = requests.get("http://localhost:8000/docs", timeout=5)
        if response.status_code == 200:
            print("✅ API is responding")
            return True
        else:
            print(f"❌ API returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

def run_tests():
    """Run the test suite"""
    print("\n🧪 Running tests...")
    
    test_files = [
        "test_main2_format.py",
        "test_frontend_backend.py"
    ]
    
    for test_file in test_files:
        if os.path.exists(test_file):
            print(f"\n🔍 Running {test_file}...")
            try:
                result = subprocess.run([sys.executable, test_file], 
                                      capture_output=True, text=True, timeout=60)
                if result.returncode == 0:
                    print(f"✅ {test_file} passed")
                else:
                    print(f"❌ {test_file} failed")
                    print(result.stdout)
                    print(result.stderr)
            except subprocess.TimeoutExpired:
                print(f"⏰ {test_file} timed out")
            except Exception as e:
                print(f"❌ {test_file} error: {e}")
        else:
            print(f"⚠️ {test_file} not found")

def show_instructions():
    """Show usage instructions"""
    print("\n" + "="*60)
    print("🎉 GROUNDWATER CHATBOT SYSTEM READY!")
    print("="*60)
    print("\n📋 Next Steps:")
    print("1. Backend is running at: http://localhost:8000")
    print("2. API documentation: http://localhost:8000/docs")
    print("3. Test the API with: python test_main2_format.py")
    print("4. Start frontend: cd frontend && npm run dev")
    print("5. Open browser to: http://localhost:5173")
    print("\n💡 Test Query: 'ground water estimation in karnataka'")
    print("\n🔧 Available Endpoints:")
    print("   • POST /ingres/query - Enhanced groundwater analysis")
    print("   • POST /ask-formatted - Simple formatted responses")
    print("   • GET /docs - API documentation")
    print("\n📊 Features:")
    print("   • Proper markdown table formatting")
    print("   • Dual vector store support (Qdrant + ChromaDB)")
    print("   • Automatic fallback between databases")
    print("   • Professional groundwater reports")
    print("\n" + "="*60)

def main():
    """Main launcher function"""
    print("🚀 Groundwater Chatbot System Launcher")
    print("="*50)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first")
        return
    
    # Check environment
    if not check_environment():
        print("\n❌ Please configure environment variables first")
        return
    
    # Start backend
    backend_process = start_backend()
    if not backend_process:
        print("\n❌ Failed to start backend")
        return
    
    # Wait for backend to start
    print("\n⏳ Waiting for backend to start...")
    time.sleep(5)
    
    # Test API
    if test_api():
        print("\n✅ System is ready!")
        
        # Run tests
        run_tests()
        
        # Show instructions
        show_instructions()
        
        # Keep running
        try:
            print("\n🔄 Backend is running. Press Ctrl+C to stop.")
            backend_process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Stopping backend...")
            backend_process.terminate()
            print("✅ Backend stopped")
    else:
        print("\n❌ Backend failed to start properly")
        backend_process.terminate()

if __name__ == "__main__":
    main()
