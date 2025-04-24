# backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import json
from utils.keyword_matcher import KeywordMatcher

app = Flask(__name__)
CORS(app)

# Load response data
with open('response_data.json', 'r', encoding='utf-8') as f:
    RESPONSE_DATA = json.load(f)

# Initialize keyword matcher
matcher = KeywordMatcher(RESPONSE_DATA)

# LibreTranslate API endpoint
LIBRE_TRANSLATE_URL = "https://libretranslate.de/translate"

def translate_text(text, source_lang, target_lang):
    """Translate text using LibreTranslate API"""
    try:
        payload = {
            "q": text,
            "source": source_lang,
            "target": target_lang,
            "format": "text"
        }
        response = requests.post(LIBRE_TRANSLATE_URL, data=payload)
        if response.status_code == 200:
            return response.json()["translatedText"]
        else:
            print(f"Translation API error: {response.text}")
            return text  # Return original text if translation fails
    except Exception as e:
        print(f"Translation error: {str(e)}")
        return text

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message')
        user_language = data.get('language', 'en')
        
        # Check if the message is already in English
        # if user_language != 'en':
        #     english_message = translate_text(user_message, user_language, 'en')
        # else:
        #     english_message = user_message
        
        # Get answer from keyword matcher using the original language
        answer = matcher.get_response(user_message, user_language)
        
        return jsonify({
            'answer': answer,
            'original_question': user_message,
            'detected_language': user_language
        })
    
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/welcome', methods=['GET'])
def welcome():
    try:
        language = request.args.get('language', 'en')
        if language not in RESPONSE_DATA['languages']:
            language = RESPONSE_DATA['default_language']
        
        welcome_message = RESPONSE_DATA['languages'][language]['greetings']['welcome']
        help_message = RESPONSE_DATA['languages'][language]['greetings']['how_can_help']
        
        return jsonify({
            'welcome': welcome_message,
            'help': help_message,
            'full_message': f"{welcome_message} {help_message}"
        })
    
    except Exception as e:
        print(f"Error in welcome endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)