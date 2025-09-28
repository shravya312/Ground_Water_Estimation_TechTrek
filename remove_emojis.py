#!/usr/bin/env python3
"""
Remove all emoji characters from main2.py
"""

import re

def remove_emojis():
    """Remove emoji characters from main2.py"""
    print("Removing emoji characters from main2.py...")
    
    # Read the file
    with open('main2.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove emoji characters
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"  # dingbats
        "\U000024C2-\U0001F251"  # enclosed characters
        "]+", flags=re.UNICODE)
    
    # Replace emojis with text equivalents
    replacements = {
        '🔄': '[INIT]',
        '✅': '[OK]',
        '❌': '[ERROR]',
        '⚠️': '[WARNING]',
        '🔍': '[DEBUG]',
        '🌧️': '[RAIN]',
        '🏖️': '[COASTAL]',
        '🏗️': '[STORAGE]'
    }
    
    # Apply replacements
    for emoji, replacement in replacements.items():
        content = content.replace(emoji, replacement)
    
    # Remove any remaining emojis
    content = emoji_pattern.sub('', content)
    
    # Write the file back
    with open('main2.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("Emoji characters removed successfully")

if __name__ == "__main__":
    remove_emojis()
