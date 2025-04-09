from openai import OpenAI
import os
os.environ["OPENAI_API_KEY"] = "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"

client = OpenAI()  # Automatically loads your OPENAI_API_KEY from env variable

def generate_answer(prompt: str, max_tokens: int = 3000) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",  # Or "gpt-3.5-turbo"
            messages=[
                {"role": "system", "content":  "You are a professional football tactics analyst. "
        "Based on the match data provided, respond with a clear, confident, and structured analysis. "
        "Avoid speculation or personal phrases like 'I think' or 'maybe'. "
        "Explain what happened in clear tactical terms using the stats, summaries, and patterns. "
        "If multiple matches are shown, identify common patterns and explain how to beat or emulate the team"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("❌ Error:", e)
        return "Sorry, there was an error generating the response."
