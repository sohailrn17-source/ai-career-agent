import os

from dotenv import load_dotenv
from google import genai

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate


# Load API key
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("ERROR: GOOGLE_API_KEY not found.")
    exit()


# Connect to Gemini
client = genai.Client(api_key=api_key)


# Load local embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Load Chroma knowledge base
vector_store = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)


# RAG prompt
rag_prompt = ChatPromptTemplate.from_template("""
You are an AI Career Agent.

Use the provided career knowledge to answer the user's question.

Rules:
- Use the knowledge provided below.
- Do not invent facts.
- If the knowledge does not contain the answer, say that the information is not available in the knowledge base.
- Give practical, beginner-friendly explanations.
- Clearly distinguish facts from recommendations.

Career knowledge:
{context}

User question:
{question}

Answer:
""")


# Ask the user
question = input("What do you want to know about your career? ")


# Retrieve relevant knowledge
results = vector_store.similarity_search(
    question,
    k=3
)


# Combine retrieved information
context = "\n\n".join(
    document.page_content for document in results
)


# Create the final prompt
final_prompt = rag_prompt.invoke({
    "context": context,
    "question": question
})


print("\nSending question to Gemini...\n")


# Send ONE request to Gemini
interaction = client.interactions.create(
    model="gemini-3.8-flash",
    input=final_prompt.to_string(),
    generation_config={
        "thinking_level": "low"
    }
)


print("AI Career Agent:")
print(interaction.output_text)