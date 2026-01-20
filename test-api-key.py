#!/usr/bin/env python3
"""Test OpenAI API Key"""
import os
from openai import OpenAI

def test_api_key():
    api_key = os.getenv('OPENAI_API_KEY', '')
    
    if not api_key:
        print("❌ ERROR: OPENAI_API_KEY environment variable is not set")
        return False
    
    print(f"✓ API Key found (length: {len(api_key)})")
    print(f"✓ Key starts with 'sk-': {api_key.startswith('sk-')}")
    print(f"✓ First 10 characters: {api_key[:10]}...")
    print("")
    print("Testing API key with OpenAI...")
    
    try:
        client = OpenAI(api_key=api_key)
        # Try a simple API call to verify the key
        models = client.models.list()
        print("✓ API Key is VALID and working!")
        print(f"✓ Successfully connected to OpenAI API")
        return True
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        if "401" in str(e) or "invalid_api_key" in str(e):
            print("")
            print("🔧 SOLUTION:")
            print("1. Go to: https://platform.openai.com/api-keys")
            print("2. Create a new API key")
            print("3. Update your .env file with the new key")
            print("4. Restart Docker: docker-compose down && docker-compose up --build")
        return False

if __name__ == "__main__":
    test_api_key()

