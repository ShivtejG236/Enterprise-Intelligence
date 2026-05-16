import os
import json
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from typing import Dict, Any, Optional

import config

def setup_gemini():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Warning: GOOGLE_API_KEY not set.")
    genai.configure(api_key=api_key)

setup_gemini()

def generate_response(prompt: str, model_name: str = config.GEMINI_CHAT_MODEL, json_mode: bool = False, temperature: float = 0.2) -> str:
    """
    Calls Gemini API and returns the text response.
    Returns a graceful error string on rate limit (429) instead of crashing.
    """
    from google.api_core.exceptions import ResourceExhausted
    model = genai.GenerativeModel(model_name)
    
    gen_config = GenerationConfig(temperature=temperature)
    if json_mode:
        gen_config.response_mime_type = "application/json"
    
    try:
        response = model.generate_content(prompt, generation_config=gen_config)
        return response.text
    except ResourceExhausted as e:
        import re
        match = re.search(r'retry in (\d+\.?\d*)s', str(e))
        wait = f" Please retry in {match.group(1)}s." if match else ""
        if json_mode:
            import json
            return json.dumps({"error": "rate_limit", "message": f"Gemini API quota exceeded.{wait}"})
        return f"⚠️ Gemini API rate limit reached.{wait}"

def generate_structured_response(prompt: str, model_name: str = config.GEMINI_REASONING_MODEL) -> Dict[str, Any]:
    """
    Calls Gemini API, requests JSON format, and parses the output.
    """
    text_resp = generate_response(prompt, model_name=model_name, json_mode=True)
    try:
        return json.loads(text_resp)
    except json.JSONDecodeError:
        # Fallback parsing if JSON wasn't strictly returned
        # Often Gemini wraps JSON in ```json ... ``` blocks
        import re
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", text_resp, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except:
                pass
        return {"error": "Failed to parse JSON", "raw": text_resp}
