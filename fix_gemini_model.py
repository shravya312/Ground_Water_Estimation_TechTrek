#!/usr/bin/env python3
"""
Fix Gemini model integration by trying different model names
"""

import os
import google.generativeai as genai

def test_gemini_models():
    """Test different Gemini model names"""
    print('🧪 Testing Gemini Model Names')
    print('=' * 40)
    
    # Set the new API key
    api_key = "AIzaSyB6NqWRoHpsRJKXZgyWxbparwd7Dk3OFQA"
    os.environ['GEMINI_API_KEY'] = api_key
    
    try:
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Try different model names
        model_names_to_try = [
            'gemini-1.5-pro',
            'gemini-1.5-flash', 
            'gemini-pro',
            'models/gemini-1.5-pro',
            'models/gemini-1.5-flash',
            'models/gemini-pro'
        ]
        
        working_model = None
        
        for model_name in model_names_to_try:
            try:
                print(f'Testing {model_name}...')
                model = genai.GenerativeModel(model_name)
                response = model.generate_content("What is the capital of India?")
                print(f'   ✅ {model_name}: Working - {response.text[:50]}...')
                if not working_model:
                    working_model = model_name
            except Exception as e:
                print(f'   ❌ {model_name}: {str(e)[:80]}...')
        
        return working_model
        
    except Exception as e:
        print(f'❌ Error testing models: {e}')
        return None

def update_main_py(model_name):
    """Update main.py with the correct model name"""
    print(f'\n🔧 Updating main.py with model: {model_name}')
    print('=' * 40)
    
    try:
        # Read main.py
        with open('main.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find and replace the model name
        import re
        
        # Look for the GenerativeModel line
        pattern = r"model = genai\.GenerativeModel\('([^']+)'\)"
        match = re.search(pattern, content)
        
        if match:
            old_model = match.group(1)
            new_content = content.replace(f"model = genai.GenerativeModel('{old_model}')", 
                                        f"model = genai.GenerativeModel('{model_name}')")
            print(f'✅ Replaced: {old_model} -> {model_name}')
        else:
            print('❌ Could not find GenerativeModel line in main.py')
            return False
        
        # Write back to file
        with open('main.py', 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print('✅ main.py updated successfully')
        return True
        
    except Exception as e:
        print(f'❌ Error updating main.py: {e}')
        return False

def test_server():
    """Test if server starts with the new model"""
    print(f'\n🚀 Testing Server with Updated Model')
    print('=' * 40)
    
    try:
        import subprocess
        import time
        import requests
        
        # Set environment variable
        env = os.environ.copy()
        env['GEMINI_API_KEY'] = "AIzaSyB6NqWRoHpsRJKXZgyWxbparwd7Dk3OFQA"
        
        # Start server
        process = subprocess.Popen(
            ['python', '-m', 'uvicorn', 'main:app', '--reload', '--port', '8000'],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for server to start
        time.sleep(8)
        
        # Test if server is responding
        try:
            response = requests.get('http://localhost:8000/ingres/states', timeout=5)
            if response.status_code == 200:
                print('✅ Server started successfully')
                
                # Test a query
                query_response = requests.post('http://localhost:8000/ingres/query', json={
                    'query': 'groundwater analysis for karnataka',
                    'include_visualizations': True
                }, timeout=10)
                
                if query_response.status_code == 200:
                    data = query_response.json()
                    print('✅ INGRES query working')
                    print(f'📊 Criticality: {data.get("criticality_status", "N/A")} {data.get("criticality_emoji", "")}')
                    print(f'📈 Visualizations: {len(data.get("visualizations", []))} charts')
                    return True
                else:
                    print(f'❌ INGRES query failed: {query_response.status_code}')
                    return False
            else:
                print(f'❌ Server not responding: {response.status_code}')
                return False
        except requests.exceptions.RequestException as e:
            print(f'❌ Server connection failed: {e}')
            return False
        finally:
            # Clean up
            try:
                process.terminate()
                process.wait(timeout=5)
            except:
                process.kill()
                
    except Exception as e:
        print(f'❌ Error testing server: {e}')
        return False

def main():
    """Main function"""
    print('🔧 Fixing Gemini API Integration')
    print('=' * 50)
    
    # Step 1: Test models
    working_model = test_gemini_models()
    
    if not working_model:
        print('❌ No working Gemini model found')
        return False
    
    # Step 2: Update main.py
    if not update_main_py(working_model):
        print('❌ Failed to update main.py')
        return False
    
    # Step 3: Test server
    if test_server():
        print('\n🎉 SUCCESS! Gemini API key is working and integrated!')
        print(f'✅ Using model: {working_model}')
        print('✅ Server starts successfully')
        print('✅ INGRES queries work with Gemini')
        return True
    else:
        print('\n❌ Server test failed')
        return False

if __name__ == "__main__":
    main()
