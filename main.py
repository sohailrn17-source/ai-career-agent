from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

client = genai.Client(api_key=api_key)

print("🤖 AI Career Agent")
print("------------------")

user_question = input("What do you want help with in your career? ")

interaction = client.interactions.create(
   model="gemini-3.7-flash",
    input=f"""
You are an AI Career Agent.

Help the user with their career, job search, resume,
skills, interviews, and learning roadmap.

Give practical and beginner-friendly advice.

User question:
{user_question}
""",
    generation_config={
        "thinking_level": "low"
    }
)

print("\nAI Career Agent:")
print(interaction.output_text)