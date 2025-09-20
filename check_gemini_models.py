#!/usr/bin/env python3
"""
Check available Gemini models and fix the integration
"""

import os
import google.generativeai as genai

def check_available_models():
    """Check what Gemini models are available"""
    print('🔍 Checking Available Gemini Models')
    print('=' * 40)
    
    # Set the new API key
    api_key = "AIzaSyB6NqWRoHpsRJKXZgyWxbparwd7Dk3OFQA"
    os.environ['GEMINI_API_KEY'] = api_key
    
    try:
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # List available models
        models = genai.list_models()
        print(f'✅ Found {len(models)} available models:')
        
        for model in models:
            print(f'   📝 {model.name}')
            if 'generateContent' in model.supported_generation_methods:
                print(f'      ✅ Supports generateContent')
            else:
                print(f'      ❌ No generateContent support')
        
        # Try different model names
        model_names_to_try = [
            'gemini-1.5-pro',
            'gemini-1.5-flash',
            'gemini-pro',
            'models/gemini-1.5-pro',
            'models/gemini-1.5-flash',
            'models/gemini-pro'
        ]
        
        print(f'\n🧪 Testing Model Names:')
        working_model = None
        
        for model_name in model_names_to_try:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content("What is 2+2?")
                print(f'   ✅ {model_name}: Working')
                if not working_model:
                    working_model = model_name
            except Exception as e:
                print(f'   ❌ {model_name}: {str(e)[:50]}...')
        
        return working_model
        
    except Exception as e:
        print(f'❌ Error checking models: {e}')
        return None

def update_main_py_with_correct_model(model_name):
    """Update main.py with the correct model name"""
    print(f'\n🔧 Updating main.py with model: {model_name}')
    print('=' * 40)
    
    try:
        # Read main.py
        with open('main.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find and replace the model name
        old_model_line = "model = genai.GenerativeModel('gemini-pro')"
        new_model_line = f"model = genai.GenerativeModel('{model_name}')"
        
        if old_model_line in content:
            content = content.replace(old_model_line, new_model_line)
            print(f'✅ Replaced: {old_model_line}')
            print(f'✅ With: {new_model_line}')
        else:
            # Try to find the line with GenerativeModel
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'GenerativeModel' in line and 'gemini-pro' in line:
                    lines[i] = new_model_line
                    content = '\n'.join(lines)
                    print(f'✅ Updated line {i+1}: {new_model_line}')
                    break
        
        # Write back to file
        with open('main.py', 'w', encoding='utf-8') as f:
            f.write(content)
        
        print('✅ main.py updated successfully')
        return True
        
    except Exception as e:
        print(f'❌ Error updating main.py: {e}')
        return False

def test_updated_integration():
    """Test the updated integration"""
    print(f'\n🧪 Testing Updated Integration')
    print('=' * 40)
    
    try:
        # Test the updated main.py
        import main
        
        if hasattr(main, '_gemini_model') and main._gemini_model:
            print('✅ Gemini model loaded in main.py')
            
            # Test a simple query
            test_coords = (28.7041, 77.1025)  # Delhi
            state = main.get_state_from_coordinates(test_coords[0], test_coords[1])
            print(f'📍 State detection test: {test_coords} -> {state}')
            
            return True
        else:
            print('❌ Gemini model still not loaded')
            return False
            
    except Exception as e:
        print(f'❌ Error testing integration: {e}')
        return False

def main():
    """Main function"""
    print('🔧 Fixing Gemini API Integration')
    print('=' * 50)
    
    # Step 1: Check available models
    working_model = check_available_models()
    
    if not working_model:
        print('❌ No working Gemini model found')
        return False
    
    # Step 2: Update main.py
    if update_main_py_with_correct_model(working_model):
        print(f'✅ Updated main.py with {working_model}')
    else:
        print('❌ Failed to update main.py')
        return False
    
    # Step 3: Test updated integration
    if test_updated_integration():
        print('✅ Integration working after update')
        return True
    else:
        print('❌ Integration still not working')
        return False

if __name__ == "__main__":
    main()
