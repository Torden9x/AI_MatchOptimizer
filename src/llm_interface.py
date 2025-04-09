from openai import OpenAI
from dotenv import load_dotenv
import os

# ✅ Load .env file
load_dotenv()

# 🔐 Get API key securely
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_answer(prompt: str, max_tokens: int = 3000) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "system", "content": (
                    "You are a professional football tactics analyst. "
                    "Based on the match data provided, respond with a clear, confident, and structured analysis. "
                    "Avoid speculation or personal phrases like 'I think' or 'maybe'. "
                    "Explain what happened in clear tactical terms using the stats, summaries, and patterns. "
                    "If multiple matches are shown, identify common patterns and explain how to beat or emulate the team."
                )},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("❌ Error:", e)
        return "Sorry, there was an error generating the response."
