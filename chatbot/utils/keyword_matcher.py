# backend/utils/keyword_matcher.py
import re
from collections import defaultdict
import logging
import unicodedata

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KeywordMatcher:
    def __init__(self, response_data):
        self.response_data = response_data
        self.keyword_maps = {}
        self._build_keyword_maps()
        
    def _build_keyword_maps(self):
        """Build keyword maps for each language"""
        for lang_code, lang_data in self.response_data['languages'].items():
            keyword_map = defaultdict(list)
            
            # Add greeting keywords
            greetings = lang_data['greetings']['greeting_responses']
            welcome_msg = lang_data['greetings']['welcome']
            help_msg = lang_data['greetings']['how_can_help']
            
            for greeting in greetings:
                keyword_map[greeting.lower()].append({
                    'id': 0,
                    'keywords': greetings,
                    'answer': f"{welcome_msg} {help_msg}"
                })
            
            # Add question keywords
            for item in lang_data['questions']:
                keywords = item['keywords']
                for keyword in keywords:
                    keyword_map[keyword.lower()].append(item)
            
            self.keyword_maps[lang_code] = keyword_map
            logger.info(f"Built keyword map for {lang_code} with {len(keyword_map)} entries")
    

    def _normalize_text(self, text, language):
        """Normalize text based on language-specific requirements"""
         # Convert to lowercase (for scripts like English)
        normalized = text.lower()
  
      # Normalize to NFC form to keep composed characters intact
        normalized = unicodedata.normalize("NFC", normalized)

    # Remove only standard ASCII punctuation (preserve all Unicode characters)
        normalized = re.sub(r'[!"#$%&\'()*+,\-./:;<=>?@[\\]^_`{|}~]', '', normalized)

    # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()

        return normalized

    
    def get_response(self, question, language='en'):
        """Get response for a question in the specified language"""
        # Default to English if language not supported
        if language not in self.keyword_maps:
            logger.warning(f"Language {language} not supported, defaulting to {self.response_data['default_language']}")
            language = self.response_data['default_language']
        
        # Normalize the question
        cleaned_question = self._normalize_text(question, language)
        
        # Log for debugging
        logger.info(f"Processing: '{question}' (normalized: '{cleaned_question}') in language: {language}")
        
        # Split into words
        words = cleaned_question.split()
        
        # First check for greeting matches
        for word in words:
            if word in self.keyword_maps[language]:
                for item in self.keyword_maps[language][word]:
                    if item['id'] == 0:  # Greeting match
                        logger.info(f"Found greeting match: {word}")
                        return item['answer']
        
        # Then check for exact keyword matches
        matches = []

        # For Tamil and Hindi, check if any keyword exists in the question FIRST
        if language in ['ta', 'hi']:
            logger.info(f"Checking for direct keyword presence in {language} question")
            # Check each keyword directly against the full question
            for keyword, items in self.keyword_maps[language].items():
                if keyword in cleaned_question:
                    logger.debug(f"Found direct match for keyword '{keyword}' in question")
                    matches.extend([item for item in items if item['id'] != 0])
        
        # If no matches found or different language, try word-by-word match
        if not matches:
            for word in words:
                if word in self.keyword_maps[language]:
                    logger.debug(f"Found exact match for word: {word}")
                    matches.extend([
                        item for item in self.keyword_maps[language][word] if item['id'] != 0
                    ])
        
        # If still no matches found for Tamil/Hindi, try more flexible matching
        if not matches and language in ['ta', 'hi']:
            logger.info(f"No direct matches found, trying partial matching for {language}")
            # Check if any word in the question is part of a keyword
            for word in words:
                if len(word) >= 3:  # Only consider words with at least 3 characters
                    for keyword in self.keyword_maps[language]:
                        if word in keyword or keyword in word:
                            logger.debug(f"Found partial match - word '{word}' related to keyword '{keyword}'")
                            matches.extend([
                                item for item in self.keyword_maps[language][keyword] if item['id'] != 0
                            ])
        
        # Process matches and return best response
        if matches:
            unique_matches = []
            seen_ids = set()
            
            # Remove duplicate matches based on ID
            for match in matches:
                if match['id'] not in seen_ids:
                    unique_matches.append(match)
                    seen_ids.add(match['id'])
            
            logger.info(f"Found {len(unique_matches)} unique matches")
            
            if len(unique_matches) == 1:
                # If only one match, return it directly
                return unique_matches[0]['answer']
            else:
                # If multiple matches, find the best one based on keyword coverage
                best_match = max(unique_matches, key=lambda x: sum(
                    1 for kw in x['keywords'] if kw.lower() in cleaned_question
                ))
                logger.info(f"Best match has ID: {best_match['id']}")
                return best_match['answer']
        else:
            logger.info("No matches found, returning default response")
            return self.response_data['languages'][language].get(
                'default_response',
                "I'm sorry, I don't have information on that topic. Please try another question."
            )
