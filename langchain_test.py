from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import os

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
   model="gemini-3.8-flash",
    google_api_key=api_key,
    temperature=0.2
)

response = llm.invoke(
    "Explain what an AI agent is in simple words."
)

print(response.content)