#!/usr/bin/env python3
"""
Check available models with new Gemini API key
"""

import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

def check_available_models():
    """Check what models are available with the new API key"""
    print("🔍 Checking Available Models with New API Key")
    print("=" * 50)
    
    new_key = "AIzaSyB_ZpSczZR8F70u7_-pXPUrtHZhOTaRrzg"
    
    try:
        # Configure with new key
        genai.configure(api_key=new_key)
        
        # List available models
        print("🔄 Listing available models...")
        models = list(genai.list_models())
        print(f"✅ Found {len(models)} models")
        
        for model in models:
            print(f"  - {model.name}")
        
        # Try different model names
        model_names_to_try = [
            'gemini-1.5-flash',
            'gemini-1.5-pro',
            'gemini-pro',
            'gemini-1.0-pro',
            'gemini-1.5-flash-002',
            'gemini-1.5-pro-002'
        ]
        
        print("\n🔄 Testing different model names...")
        for model_name in model_names_to_try:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content("Hello")
                print(f"✅ {model_name}: Working - {response.text[:50]}...")
            except Exception as e:
                print(f"❌ {model_name}: {str(e)[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    check_available_models()
